//! Builds the edit `cargo dylint --fix` applies: the part moves into a new free fn after the
//! enclosing item and a call takes its place. Refused unless the result is certain to compile.

mod context;
mod print;
mod run;
mod signature;

use std::ops::ControlFlow;

use clippy_utils::source::{snippet_indent, snippet_opt};
use clippy_utils::visitors::for_each_expr;
use rustc_hir::def::Res;
use rustc_hir::def_id::{LocalDefId, LocalModDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{
    BindingMode, ByRef, Expr, ExprKind, HirId, ItemKind, Lit, Mutability, Pat, PatKind, QPath,
    StmtKind, UnOp,
};
use rustc_lint::LateContext;
use rustc_middle::mir;
use rustc_middle::ty::adjustment::{Adjust, AutoBorrow, AutoBorrowMutability, DerefAdjustKind};
use rustc_middle::ty::{TyCtxt, TypeckResults};
use rustc_span::source_map::SourceMap;
use rustc_span::{BytePos, Ident, Pos, Span, Symbol};

use crate::generic_body_not_generic::region::SharedPart;
use crate::generic_body_not_generic::source_span::SourceSpan;
use context::{MoveContext, movable_context};
use run::{Run, matched_run};
use signature::{CallResult, CallSignature, call_signature};

/// One machine-applicable suggestion: non-overlapping (span, replacement) parts.
pub(super) struct Edit {
    pub(super) help: String,
    pub(super) parts: Vec<(Span, String)>,
}

pub(super) fn extraction_edit<'tcx>(
    cx: &LateContext<'tcx>,
    def: LocalDefId,
    body: &mir::Body<'tcx>,
    part: &SharedPart,
    site: Option<SourceSpan>,
) -> Option<Edit> {
    let tcx = cx.tcx;
    let run = matched_run(tcx, def, body, part, site)?;
    let context = movable_context(tcx, def, body, part, &run)?;
    let signature = call_signature(cx, def, body, part, &run)?;
    let name = free_name(
        tcx,
        context.module,
        tcx.opt_item_name(def.to_def_id())?.as_str(),
    )?;
    assemble(cx, def, &run, &context, &signature, &name)
}

/// `<base>_shared`, unless `module` has that name: an item, an `extern` item or any binding.
fn free_name(tcx: TyCtxt<'_>, module: LocalModDefId, base: &str) -> Option<String> {
    let name = format!("{base}_shared");
    let taken = Symbol::intern(&name);
    let resolved = tcx
        .module_children_local(module.to_local_def_id())
        .iter()
        .any(|child| child.ident.name == taken);
    let (items, _, _) = tcx.hir_get_module(module);
    let written = items.item_ids.iter().any(|&id| {
        let item = tcx.hir_item(id);
        match item.kind {
            ItemKind::ForeignMod { items, .. } => items
                .iter()
                .any(|&foreign| tcx.hir_foreign_item(foreign).ident.name == taken),
            kind => kind.ident().is_some_and(|ident| ident.name == taken),
        }
    });
    (!resolved && !written).then_some(name)
}

/// `mutated`: bindings the run assigns or borrows exclusively, not through a pointer. `all`: a
/// `ref mut` pattern, keep every `let mut`. `verbatim`: a literal spans lines, keep indentation.
struct MovedText<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    typeck: &'tcx TypeckResults<'tcx>,
    source_map: &'a SourceMap,
    mutated: Vec<HirId>,
    all: bool,
    verbatim: bool,
}

impl MovedText<'_, '_> {
    /// `expr`'s first adjustment: `Some(true)` borrows it exclusively, `None` leaves the place.
    fn adjusted(&self, expr: &Expr<'_>) -> Option<bool> {
        let mut ty = self.typeck.expr_ty_opt(expr)?;
        for adjust in self.typeck.expr_adjustments(expr) {
            match adjust.kind {
                Adjust::NeverToAny | Adjust::Pointer(_) => {}
                Adjust::Deref(DerefAdjustKind::Builtin) if ty.is_box() => {}
                Adjust::Deref(DerefAdjustKind::Overloaded(deref)) => {
                    return Some(deref.mutbl.is_mut());
                }
                Adjust::Borrow(
                    AutoBorrow::Ref(AutoBorrowMutability::Mut { .. })
                    | AutoBorrow::RawPtr(Mutability::Mut)
                    | AutoBorrow::Pin(Mutability::Mut),
                )
                | Adjust::GenericReborrow(Mutability::Mut) => return Some(true),
                _ => return Some(false),
            }
            ty = adjust.target;
        }
        None
    }

    fn mark(&mut self, mut expr: &Expr<'_>) {
        loop {
            expr = match expr.kind {
                ExprKind::Field(base, _) | ExprKind::Index(base, _, _)
                    if self.adjusted(base).is_none() =>
                {
                    base
                }
                ExprKind::Unary(UnOp::Deref, inner)
                    if self
                        .typeck
                        .expr_ty_adjusted_opt(inner)
                        .is_some_and(|ty| ty.is_box()) =>
                {
                    inner
                }
                ExprKind::Path(QPath::Resolved(None, path)) => {
                    if let Res::Local(id) = path.res {
                        self.mutated.push(id);
                    }
                    return;
                }
                _ => return,
            };
        }
    }
}

impl<'tcx> Visitor<'tcx> for MovedText<'_, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        if self.adjusted(expr) == Some(true) {
            self.mark(expr);
        }
        match expr.kind {
            ExprKind::Assign(place, _, _)
            | ExprKind::AssignOp(_, place, _)
            | ExprKind::AddrOf(_, Mutability::Mut, place) => self.mark(place),
            ExprKind::Closure(closure) => {
                intravisit::walk_body(self, self.tcx.hir_body(closure.body));
            }
            _ => {}
        }
        intravisit::walk_expr(self, expr);
    }

    fn visit_lit(&mut self, _: HirId, lit: Lit, _: bool) {
        self.verbatim |= self.source_map.is_multiline(lit.span.source_callsite());
    }

    fn visit_pat(&mut self, pat: &'tcx Pat<'tcx>) {
        self.all |= matches!(
            pat.kind,
            PatKind::Binding(BindingMode(ByRef::Yes(_, Mutability::Mut), _), ..)
        );
        intravisit::walk_pat(self, pat);
    }
}

/// The call replacing the run, and `fn name` after the item. `None`: a hidden `let` is read later.
fn assemble<'tcx>(
    cx: &LateContext<'tcx>,
    def: LocalDefId,
    run: &Run<'tcx>,
    context: &MoveContext,
    signature: &CallSignature,
    name: &str,
) -> Option<Edit> {
    let tcx = cx.tcx;
    let stmts = &run.block.stmts[run.stmts.clone()];
    let mut moved = snippet_opt(cx, run.span)?;
    let mut text = MovedText {
        tcx,
        typeck: tcx.typeck(def),
        source_map: tcx.sess.source_map(),
        mutated: Vec::new(),
        all: false,
        verbatim: false,
    };
    for stmt in stmts {
        text.visit_stmt(stmt);
    }
    if run.with_tail
        && let Some(tail) = run.block.expr
    {
        text.visit_expr(tail);
    }

    let produced: &[(Symbol, bool)] = match &signature.result {
        CallResult::Lets { names, .. } => names,
        CallResult::Unit | CallResult::Tail(_) => &[],
    };
    let mut lost: Vec<HirId> = Vec::new();
    for stmt in stmts.iter().rev() {
        let StmtKind::Let(let_) = stmt.kind else {
            continue;
        };
        let_.pat.each_binding(|_, id, _, ident| {
            if !produced.iter().any(|&(var, _)| var == ident.name) {
                lost.push(id);
            }
        });
        // `let mut x` that the run never changes: a plain `let` in the new fn.
        let offset = |at: BytePos| Pos::to_usize(&(at - run.span.lo()));
        if let PatKind::Binding(BindingMode(ByRef::No, Mutability::Mut), id, ident, None) =
            let_.pat.kind
            && !text.all
            && !text.mutated.contains(&id)
            && run.span.contains(let_.pat.span)
            && let_.pat.span.eq_ctxt(run.span)
            && ident.span.eq_ctxt(run.span)
            && let cut = (offset(let_.pat.span.lo())..offset(ident.span.lo()))
            && moved
                .get(cut.clone())
                .is_some_and(|word| word.trim_end() == "mut")
        {
            moved.replace_range(cut, "");
        }
    }
    let named_after = for_each_expr(cx, tcx.hir_body_owned_by(def).value, |expr| {
        if let ExprKind::Path(QPath::Resolved(None, path)) = expr.kind
            && let Res::Local(id) = path.res
            && !run.span.contains(expr.span)
            && lost.contains(&id)
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    if named_after.is_some() {
        return None;
    }

    let outer = snippet_indent(cx, context.insert_at).unwrap_or_default();
    let old = snippet_indent(cx, run.span).unwrap_or_default();
    let step = if outer.contains('\t') || old.contains('\t') {
        "\t"
    } else {
        "    "
    };
    let inner = if text.verbatim {
        old.clone()
    } else {
        format!("{outer}{step}")
    };
    let mut body = String::new();
    for (i, line) in moved.split('\n').enumerate() {
        let rest = match i {
            0 => Some(line),
            _ if text.verbatim => None,
            _ => line.strip_prefix(old.as_str()),
        };
        if i > 0 {
            body.push('\n');
        }
        match rest {
            Some(rest) if !rest.trim().is_empty() => {
                body.push_str(&inner);
                body.push_str(rest);
            }
            Some(rest) => body.push_str(rest),
            None => body.push_str(line),
        }
    }

    let names = |list: &[String]| match list {
        [one] => one.clone(),
        list => format!("({})", list.join(", ")),
    };
    // A name as written: `r#type` for a keyword in this edition.
    let word = |name: Symbol| Ident::new(name, run.span).to_string();
    let args: Vec<String> = signature
        .inputs
        .iter()
        .map(|&(arg, _, _)| word(arg))
        .collect();
    let call = format!("{name}({})", args.join(", "));
    let (replacement, returns) = match &signature.result {
        CallResult::Unit if run.with_tail => (call, String::new()),
        CallResult::Unit => (format!("{call};"), String::new()),
        CallResult::Lets { names: lets, tys } if !run.with_tail => {
            let pattern: Vec<String> = lets
                .iter()
                .map(|&(var, mutable)| {
                    format!("{}{}", if mutable { "mut " } else { "" }, word(var))
                })
                .collect();
            let vars: Vec<String> = lets.iter().map(|&(var, _)| word(var)).collect();
            body.push('\n');
            body.push_str(&inner);
            body.push_str(&names(&vars));
            (
                format!("let {} = {call};", names(&pattern)),
                format!(" -> {}", names(&tys[..])),
            )
        }
        CallResult::Tail(ty) if run.with_tail => (
            call,
            ty.as_ref()
                .map_or_else(String::new, |ty| format!(" -> {ty}")),
        ),
        CallResult::Lets { .. } | CallResult::Tail(_) => return None,
    };
    let params: Vec<String> = signature
        .inputs
        .iter()
        .map(|&(arg, mutable, ref ty)| {
            format!("{}{}: {ty}", if mutable { "mut " } else { "" }, word(arg))
        })
        .collect();
    let header = format!(
        "{}fn {name}({}){returns}",
        if context.const_fn { "const " } else { "" },
        params.join(", "),
    );
    Some(Edit {
        help: format!(
            "`cargo dylint --fix` moves them into a new function `{name}` and calls it here. Rename it afterwards"
        ),
        parts: vec![
            (run.span, replacement),
            (
                context.insert_at,
                format!("\n\n{outer}{header} {{\n{body}\n{outer}}}"),
            ),
        ],
    })
}
