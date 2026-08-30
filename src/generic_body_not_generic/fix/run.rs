//! Finds the run of whole statements whose text is the shared part.

use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Block, BlockCheckMode, Stmt, StmtKind};
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::{
    self, BasicBlock, Const, ConstValue, Operand, Rvalue, Statement, StatementKind,
};
use rustc_middle::ty::TyCtxt;
use rustc_span::{BytePos, Span};

use crate::generic_body_not_generic::classify::{counted_statement, counted_terminator};
use crate::generic_body_not_generic::region::SharedPart;
use crate::generic_body_not_generic::source_span::{SourceSpan, body_position};

/// The part as source text: `block.stmts[stmts]`, then `block.expr` when `with_tail`.
pub(super) struct Run<'tcx> {
    pub(super) block: &'tcx Block<'tcx>,
    pub(super) stmts: std::ops::Range<usize>,
    pub(super) with_tail: bool,
    pub(super) span: Span,
}

struct InnermostBlock<'tcx> {
    span: Span,
    found: Option<&'tcx Block<'tcx>>,
    in_unsafe: bool,
}

impl<'tcx> Visitor<'tcx> for InnermostBlock<'tcx> {
    fn visit_block(&mut self, block: &'tcx Block<'tcx>) {
        if block.span.contains(self.span) {
            self.found = Some(block);
            self.in_unsafe |= matches!(block.rules, BlockCheckMode::UnsafeBlock(_));
            intravisit::walk_block(self, block);
        }
    }
}

/// Whether a MIR item of `stmt` can start at `at`: it, its expression, pattern or initializer.
fn stmt_start(stmt: &Stmt<'_>, outer: Span, at: BytePos) -> bool {
    let inner = match stmt.kind {
        StmtKind::Let(local) => [Some(local.pat.span), local.init.map(|e| e.span)],
        StmtKind::Expr(expr) | StmtKind::Semi(expr) => [Some(expr.span), None],
        StmtKind::Item(_) => return false,
    };
    std::iter::once(stmt.span)
        .chain(inner.into_iter().flatten())
        .filter_map(|span| span.find_ancestor_inside_same_ctxt(outer))
        .any(|span| span.lo() == at)
}

/// The run of whole statements that `site` marks, if every counted MIR item agrees (`exact`).
pub(super) fn matched_run<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: LocalDefId,
    body: &mir::Body<'tcx>,
    part: &SharedPart,
    site: Option<SourceSpan>,
) -> Option<Run<'tcx>> {
    let Some(SourceSpan::Whole(site)) = site else {
        return None;
    };
    let body_expr = tcx.hir_body_owned_by(def).value;
    let mut search = InnermostBlock {
        span: site,
        found: None,
        in_unsafe: false,
    };
    search.visit_expr(body_expr);
    let block = search.found?;
    let outer = block.span;
    if search.in_unsafe || outer.from_expansion() || !outer.eq_ctxt(body_expr.span) {
        return None;
    }
    // Position `n` is `block.expr`. `site` can end inside an `if` or `match`: no MIR item spans it.
    let n = block.stmts.len();
    let tail = block
        .expr
        .map(|e| e.span)
        .filter(|s| !s.from_expansion() && s.eq_ctxt(outer));
    let start = |at: usize| match block.stmts.get(at) {
        Some(stmt) => stmt.span.find_ancestor_inside_same_ctxt(outer),
        None => tail,
    };
    let first = (0..n)
        .find(|&i| stmt_start(&block.stmts[i], outer, site.lo()))
        .or_else(|| (tail?.lo() == site.lo()).then_some(n))?;
    let last = (first..=n)
        .take_while(|&at| start(at).is_some_and(|s| s.lo() < site.hi()))
        .last()?;
    let stmts = first..(last + 1).min(n);
    for stmt in &block.stmts[stmts.clone()] {
        let refused = match stmt.kind {
            StmtKind::Item(_) => true,
            StmtKind::Let(local) => local.super_.is_some(),
            StmtKind::Expr(_) | StmtKind::Semi(_) => false,
        };
        if refused || stmt.span.from_expansion() || !stmt.span.eq_ctxt(outer) {
            return None;
        }
    }
    // An attribute on the first statement precedes its span and would stay behind on the call.
    let (first_id, lo) = match block.stmts.get(first) {
        Some(stmt) => (stmt.hir_id, stmt.span.lo()),
        None => (block.expr?.hir_id, tail?.lo()),
    };
    if !tcx.hir_attrs(first_id).is_empty() {
        return None;
    }
    let hi = match block.stmts.get(last) {
        Some(stmt) => stmt.span.hi(),
        None => tail?.hi(),
    };
    let span = outer.with_lo(lo).with_hi(hi);
    exact(body, &part.blocks, body_expr.span, span).then_some(Run {
        block,
        stmts,
        with_tail: last == n,
        span,
    })
}

/// Whether each counted item of a part block is in `run` and no other block's is, unless it
/// encloses `run`. `x = const ()` is skipped: a `while` ending the part writes it after the exit.
fn exact(body: &mir::Body<'_>, part: &DenseBitSet<BasicBlock>, body_span: Span, run: Span) -> bool {
    let unit_value = |statement: &Statement<'_>| {
        if let StatementKind::Assign(assign) = &statement.kind
            && let Rvalue::Use(Operand::Constant(constant), _) = &assign.1
            && let Const::Val(ConstValue::ZeroSized, ty) = constant.const_
        {
            ty.is_unit()
        } else {
            false
        }
    };
    body.basic_blocks.iter_enumerated().all(|(block, data)| {
        if data.is_cleanup {
            return true;
        }
        let in_part = part.contains(block);
        let statements = data
            .statements
            .iter()
            .filter(|s| counted_statement(s) && !unit_value(s))
            .map(|s| s.source_info.span);
        let terminator = data
            .terminator
            .iter()
            .filter(|t| counted_terminator(t))
            .map(|t| t.source_info.span);
        statements
            .chain(terminator)
            .all(|item| match body_position(item, body_span) {
                None => !in_part,
                Some(at) if in_part => run.contains(at),
                Some(at) => !at.overlaps(run) || (at.contains(run) && !at.source_equal(run)),
            })
    })
}
