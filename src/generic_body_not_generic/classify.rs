//! Counts the statements in each piece of the body and marks those that use a
//! generic parameter. Also finds locals whose value is a constant in each copy.

use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::{NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{
    self, BasicBlock, Local, Location, Operand, Place, Rvalue, Statement, StatementKind,
    Terminator, TerminatorKind,
};
use rustc_middle::ty::{TyCtxt, TypeVisitableExt};
use rustc_span::{ExpnKind, Span, Spanned};

use crate::mir_flow::{FlowGraph, reads_any};

/// For one basic block: items that do something at run time, those among
/// them that use a parameter ("dependent"), and those written by hand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct BlockFacts {
    pub(super) counted: u32,
    pub(super) dependent: u32,
    pub(super) hand_written: u32,
}

/// Not produced by a macro. A desugaring of hand-written code counts.
pub(super) fn hand_written(span: Span) -> bool {
    let mut ctxt = span.ctxt();
    loop {
        if ctxt.is_root() {
            return true;
        }
        let expn = ctxt.outer_expn_data();
        match expn.kind {
            ExpnKind::Root => return true,
            ExpnKind::Macro(..) => return false,
            ExpnKind::Desugaring(_) | ExpnKind::AstPass(_) => ctxt = expn.call_site.ctxt(),
        }
    }
}

pub(super) fn counted_statement(statement: &Statement<'_>) -> bool {
    !matches!(
        statement.kind,
        StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Nop
            | StatementKind::FakeRead(_)
            | StatementKind::PlaceMention(_)
            | StatementKind::AscribeUserType(..)
            | StatementKind::Coverage(_)
            | StatementKind::ConstEvalCounter
            | StatementKind::BackwardIncompatibleDropHint { .. }
    )
}

pub(super) fn counted_terminator(terminator: &Terminator<'_>) -> bool {
    !matches!(
        terminator.kind,
        TerminatorKind::Goto { .. }
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
    )
}

/// Finds whether an item uses a place whose type contains a parameter. Only
/// the final projected type is tested: `(*_1).header: [u8; 4]` does not.
pub(super) struct PlaceParams<'a, 'tcx> {
    pub(super) tcx: TyCtxt<'tcx>,
    pub(super) body: &'a mir::Body<'tcx>,
    pub(super) found: bool,
}

impl<'tcx> Visitor<'tcx> for PlaceParams<'_, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, _: PlaceContext, _: Location) {
        if place
            .ty(&self.body.local_decls, self.tcx)
            .ty
            .has_non_region_param()
        {
            self.found = true;
        }
    }
}

impl<'tcx> PlaceParams<'_, 'tcx> {
    fn statement_params(&mut self, statement: &Statement<'tcx>) -> (bool, bool) {
        self.found = false;
        self.visit_statement(statement, Location::START);
        let names_param = promoted_load_names_param(self.tcx, self.body, statement)
            .unwrap_or_else(|| statement.has_non_region_param());
        (names_param, self.found)
    }

    fn terminator_params(&mut self, terminator: &Terminator<'tcx>) -> (bool, bool) {
        self.found = false;
        self.visit_terminator(terminator, Location::START);
        (terminator.has_non_region_param(), self.found)
    }

    pub(super) fn dependent_statement(&mut self, statement: &Statement<'tcx>) -> bool {
        let (names_param, typed_place) = self.statement_params(statement);
        names_param || typed_place
    }

    pub(super) fn dependent_terminator(&mut self, terminator: &Terminator<'tcx>) -> bool {
        let (names_param, typed_place) = self.terminator_params(terminator);
        names_param || typed_place
    }

    /// Names a parameter only as a constant, a callee's generic argument or
    /// a cast type: a value that is a compile-time constant in each copy.
    fn const_only(&mut self, rhs: AssignedValue<'_, 'tcx>) -> bool {
        let (names_param, typed_place) = match rhs {
            AssignedValue::Value(statement, _) => self.statement_params(statement),
            AssignedValue::Call(terminator, ..) => self.terminator_params(terminator),
        };
        names_param && !typed_place
    }
}

/// For `_n = const promoted[k]` of this body, whether the promoted constant
/// uses a parameter. The statement itself always names every parameter.
fn promoted_load_names_param<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    statement: &Statement<'tcx>,
) -> Option<bool> {
    let (_, mir::Rvalue::Use(mir::Operand::Constant(constant), _)) = statement.kind.as_assign()?
    else {
        return None;
    };
    let mir::Const::Unevaluated(unevaluated, ty) = constant.const_ else {
        return None;
    };
    let index = unevaluated.promoted?;
    let def = body.source.def_id();
    if unevaluated.def != def {
        return None;
    }
    // `promoted_mir` does not steal the body `mir_for` reads.
    let promoted = tcx.promoted_mir(def).get(index)?;
    Some(
        ty.has_non_region_param()
            || promoted
                .local_decls
                .iter()
                .any(|local| local.ty.has_non_region_param())
            || promoted.basic_blocks.iter().any(|data| {
                data.statements.iter().any(|s| s.has_non_region_param())
                    || data
                        .terminator
                        .as_ref()
                        .is_some_and(|t| t.has_non_region_param())
            }),
    )
}

pub(super) fn classify<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
) -> (IndexVec<BasicBlock, BlockFacts>, usize) {
    let mut places = PlaceParams {
        tcx,
        body,
        found: false,
    };
    let facts: IndexVec<BasicBlock, BlockFacts> = body
        .basic_blocks
        .iter()
        .map(|data| {
            if data.is_cleanup {
                return BlockFacts::default();
            }
            let mut facts = BlockFacts::default();
            let mut count = |dependent: bool, span: Span| {
                facts.counted += 1;
                if dependent {
                    facts.dependent += 1;
                }
                if hand_written(span) {
                    facts.hand_written += 1;
                }
            };
            for statement in &data.statements {
                if !counted_statement(statement) {
                    continue;
                }
                count(
                    places.dependent_statement(statement),
                    statement.source_info.span,
                );
            }
            if let Some(terminator) = &data.terminator
                && counted_terminator(terminator)
            {
                count(
                    places.dependent_terminator(terminator),
                    terminator.source_info.span,
                );
            }
            facts
        })
        .collect();
    let total = facts.iter().map(|block| block.counted as usize).sum();
    (facts, total)
}

/// Locals whose value is a compile-time constant in each copy
/// (`size_of::<T>()`, `width_of(TAG)`), and `blocks_under_const_branch`.
pub(super) fn per_copy_consts<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    flow: &FlowGraph,
) -> (DenseBitSet<Local>, DenseBitSet<BasicBlock>) {
    let mut places = PlaceParams {
        tcx,
        body,
        found: false,
    };
    let mut consts = DenseBitSet::new_empty(body.local_decls.len());
    for (dest, rhs) in assignments(body, body.basic_blocks.indices()) {
        if let Some(local) = assigned_local(dest)
            && places.const_only(rhs)
        {
            consts.insert(local);
        }
    }
    loop {
        if !consts.is_empty() {
            add_locals_computed_from(
                body,
                || body.basic_blocks.indices(),
                &mut consts,
                |dest| assigned_local(dest).is_some(),
            );
        }
        let branch_blocks = blocks_under_const_branch(body, flow, &consts);
        if !consts_assigned_under_const_branch(body, &branch_blocks, &mut consts) {
            return (consts, branch_blocks);
        }
    }
}

/// Adds locals assigned a value that reads no local (`match TAG { 0 => 1, _ => 4 }`).
fn consts_assigned_under_const_branch(
    body: &mir::Body<'_>,
    blocks_under_const_branch: &DenseBitSet<BasicBlock>,
    per_copy_consts: &mut DenseBitSet<Local>,
) -> bool {
    let every = DenseBitSet::new_filled(body.local_decls.len());
    let mut added = false;
    for (dest, rhs) in assignments(body, blocks_under_const_branch.iter()) {
        if let Some(local) = assigned_local(dest)
            && !per_copy_consts.contains(local)
            && !matches!(
                rhs,
                AssignedValue::Value(_, Rvalue::Use(Operand::RuntimeChecks(_), _))
            )
            && !rhs.reads_any(&every)
        {
            added |= per_copy_consts.insert(local);
        }
    }
    added
}

/// Until nothing changes. Destination places are not searched.
pub(super) fn add_locals_computed_from<'tcx, I: Iterator<Item = BasicBlock>>(
    body: &mir::Body<'tcx>,
    blocks: impl Fn() -> I,
    set: &mut DenseBitSet<Local>,
    accept: impl Fn(Place<'tcx>) -> bool,
) {
    loop {
        let mut added = false;
        for (dest, rhs) in assignments(body, blocks()) {
            if !set.contains(dest.local) && accept(dest) && rhs.reads_any(set) {
                added |= set.insert(dest.local);
            }
        }
        if !added {
            return;
        }
    }
}

/// `None` for a store through a pointer or into an array element.
fn assigned_local(place: Place<'_>) -> Option<Local> {
    place
        .projection
        .iter()
        .all(|elem| {
            !matches!(
                elem,
                mir::ProjectionElem::Deref
                    | mir::ProjectionElem::Index(_)
                    | mir::ProjectionElem::ConstantIndex { .. }
                    | mir::ProjectionElem::Subslice { .. }
            )
        })
        .then_some(place.local)
}

#[derive(Clone, Copy)]
enum AssignedValue<'a, 'tcx> {
    Value(&'a Statement<'tcx>, &'a Rvalue<'tcx>),
    Call(
        &'a Terminator<'tcx>,
        &'a Operand<'tcx>,
        &'a [Spanned<Operand<'tcx>>],
    ),
}

impl<'tcx> AssignedValue<'_, 'tcx> {
    fn visit(self, visitor: &mut impl Visitor<'tcx>) {
        match self {
            AssignedValue::Value(_, rvalue) => visitor.visit_rvalue(rvalue, Location::START),
            AssignedValue::Call(_, func, args) => {
                visitor.visit_operand(func, Location::START);
                for arg in args {
                    visitor.visit_operand(&arg.node, Location::START);
                }
            }
        }
    }

    fn reads_any(self, of: &DenseBitSet<Local>) -> bool {
        reads_any(of, |uses| self.visit(uses))
    }
}

fn assignments<'a, 'tcx>(
    body: &'a mir::Body<'tcx>,
    blocks: impl Iterator<Item = BasicBlock>,
) -> impl Iterator<Item = (Place<'tcx>, AssignedValue<'a, 'tcx>)> {
    blocks
        .map(|block| &body.basic_blocks[block])
        .filter(|data| !data.is_cleanup)
        .flat_map(|data| {
            let assigns = data.statements.iter().filter_map(|statement| {
                let StatementKind::Assign(assign) = &statement.kind else {
                    return None;
                };
                Some((assign.0, AssignedValue::Value(statement, &assign.1)))
            });
            let call = data.terminator.as_ref().and_then(|terminator| {
                let TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    ..
                } = &terminator.kind
                else {
                    return None;
                };
                Some((*destination, AssignedValue::Call(terminator, func, args)))
            });
            assigns.chain(call)
        })
}

/// The blocks that run only when a branch on a per-copy constant (`if
/// FLAG`) goes one way. Each copy keeps only its own side.
pub(super) fn blocks_under_const_branch(
    body: &mir::Body<'_>,
    flow: &FlowGraph,
    per_copy_consts: &DenseBitSet<Local>,
) -> DenseBitSet<BasicBlock> {
    let per_copy = |discr: &Operand<'_>| match discr {
        Operand::Constant(constant) => constant.const_.has_non_region_param(),
        Operand::Copy(place) | Operand::Move(place) => reads_any(per_copy_consts, |uses| {
            uses.visit_place(
                place,
                PlaceContext::NonMutatingUse(NonMutatingUseContext::Inspect),
                Location::START,
            );
        }),
        Operand::RuntimeChecks(_) => false,
    };
    let mut decided = DenseBitSet::new_empty(body.basic_blocks.len());
    let mut pending: Vec<BasicBlock> = body
        .basic_blocks
        .iter_enumerated()
        .filter(|&(block, data)| {
            flow.contains(block)
                && matches!(
                    &data.terminator,
                    Some(Terminator { kind: TerminatorKind::SwitchInt { discr, .. }, .. })
                        if per_copy(discr)
                )
        })
        .map(|(block, _)| block)
        .collect();
    let mut expanded = DenseBitSet::new_empty(body.basic_blocks.len());
    while let Some(branch) = pending.pop() {
        if !expanded.insert(branch) {
            continue;
        }
        for block in flow.decides(branch) {
            if decided.insert(block) && flow.succs(block).len() >= 2 {
                pending.push(block);
            }
        }
    }
    decided
}
