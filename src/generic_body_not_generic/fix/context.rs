//! Whether the run's text can be the body of a new free fn, and where that fn goes.

use std::ops::ControlFlow::{self, Break, Continue};

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId, LocalModDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{
    Expr, ExprKind, HirId, ItemKind, Lifetime, LifetimeKind, Node, Pat, PatKind, Path, QPath, Stmt,
    StmtKind, find_attr,
};
use rustc_middle::mir;
use rustc_middle::ty::{TyCtxt, TypeckResults};
use rustc_span::symbol::kw;
use rustc_span::{Span, Symbol};

use super::run::Run;
use super::signature::user_var;
use crate::generic_body_not_generic::region::SharedPart;

/// Where the new fn goes, directly inside `module`, and how its header starts.
pub(super) struct MoveContext {
    pub(super) insert_at: Span,
    pub(super) const_fn: bool,
    pub(super) module: LocalModDefId,
}

/// The locals `run` names but does not bind. `None` for `return`, `?`, `asm!`, a jump out,
/// `self`, a generic, a named lifetime, an outside capture, an `of_trait` item only `impl` scopes.
fn run_is_movable<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: LocalDefId,
    of_trait: Option<DefId>,
    run: &Run<'tcx>,
) -> Option<Vec<HirId>> {
    struct Walk<'tcx> {
        tcx: TyCtxt<'tcx>,
        typeck: &'tcx TypeckResults<'tcx>,
        of_trait: Option<DefId>,
        targets: FxHashSet<HirId>,
        bound: FxHashSet<HirId>,
        named: FxHashSet<HirId>,
        closures: u32,
    }
    impl Walk<'_> {
        /// `item` is `of_trait`'s and in scope only by the `impl` (listed once, not per `use`).
        fn needs_impl_trait(&self, item: Option<DefId>, at: HirId) -> bool {
            let Some(of_trait) = self.of_trait else {
                return false;
            };
            let Some(item) = item else {
                return true;
            };
            if self.tcx.trait_of_assoc(item) != Some(of_trait) {
                return false;
            }
            let listed = self
                .tcx
                .in_scope_traits(at)
                .map_or(0, |all| all.iter().filter(|c| c.def_id == of_trait).count());
            listed < 2
        }
    }
    impl<'tcx> Visitor<'tcx> for Walk<'tcx> {
        type NestedFilter = rustc_middle::hir::nested_filter::OnlyBodies;
        type Result = ControlFlow<()>;
        fn maybe_tcx(&mut self) -> TyCtxt<'tcx> {
            self.tcx
        }
        fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) -> ControlFlow<()> {
            match expr.kind {
                ExprKind::Ret(_) | ExprKind::Become(_) | ExprKind::Yield(..)
                    if self.closures == 0 =>
                {
                    return Break(());
                }
                ExprKind::InlineAsm(_) => return Break(()),
                ExprKind::Break(to, _) | ExprKind::Continue(to) => {
                    if !to.target_id.is_ok_and(|id| self.targets.contains(&id)) {
                        return Break(());
                    }
                }
                ExprKind::Loop(block, ..) | ExprKind::Block(block, _) => {
                    self.targets.extend([expr.hir_id, block.hir_id]);
                }
                ExprKind::Closure(closure) => {
                    let upvars = self.tcx.upvars_mentioned(closure.def_id);
                    if upvars.is_some_and(|u| u.keys().any(|id| !self.bound.contains(id))) {
                        return Break(());
                    }
                    self.closures += 1;
                    let result = intravisit::walk_expr(self, expr);
                    self.closures -= 1;
                    return result;
                }
                ExprKind::MethodCall(..) => {
                    let method = self.typeck.type_dependent_def_id(expr.hir_id);
                    if self.needs_impl_trait(method, expr.hir_id) {
                        return Break(());
                    }
                }
                _ => {}
            }
            intravisit::walk_expr(self, expr)
        }
        fn visit_qpath(
            &mut self,
            qpath: &'tcx QPath<'tcx>,
            id: HirId,
            _span: Span,
        ) -> ControlFlow<()> {
            if let QPath::TypeRelative(..) = qpath
                && self.needs_impl_trait(self.typeck.type_dependent_def_id(id), id)
            {
                return Break(());
            }
            intravisit::walk_qpath(self, qpath, id)
        }
        fn visit_pat(&mut self, pat: &'tcx Pat<'tcx>) -> ControlFlow<()> {
            if let PatKind::Binding(_, id, ..) = pat.kind {
                self.bound.insert(id);
            }
            intravisit::walk_pat(self, pat)
        }
        fn visit_path(&mut self, path: &Path<'tcx>, _: HirId) -> ControlFlow<()> {
            if let Res::Local(id) = path.res {
                self.named.insert(id);
            }
            let generic = matches!(
                path.res,
                Res::SelfTyAlias { .. }
                    | Res::SelfTyParam { .. }
                    | Res::SelfCtor(_)
                    | Res::Def(DefKind::TyParam | DefKind::ConstParam, _)
            );
            let self_word = path
                .segments
                .first()
                .is_some_and(|s| s.ident.name == kw::SelfLower || s.ident.name == kw::SelfUpper);
            if generic || self_word {
                return Break(());
            }
            intravisit::walk_path(self, path)
        }
        fn visit_lifetime(&mut self, lifetime: &'tcx Lifetime) -> ControlFlow<()> {
            if let LifetimeKind::Param(_) = lifetime.kind {
                Break(())
            } else {
                Continue(())
            }
        }
    }

    let mut walk = Walk {
        tcx,
        typeck: tcx.typeck(def),
        of_trait,
        targets: FxHashSet::default(),
        bound: FxHashSet::default(),
        named: FxHashSet::default(),
        closures: 0,
    };
    let tail = if run.with_tail { run.block.expr } else { None };
    let movable = run.block.stmts[run.stmts.clone()]
        .iter()
        .all(|stmt| walk.visit_stmt(stmt).is_continue())
        && tail.is_none_or(|expr| walk.visit_expr(expr).is_continue());
    movable.then(|| walk.named.difference(&walk.bound).copied().collect())
}

/// Where a movable `run` can go: after the enclosing item, which must sit directly in a module.
/// Refused: an `unsafe fn`, a `cfg` or lint attribute between the run and the item, an item from
/// a macro, an item statement in the body, another generic fn of this name under the module.
pub(super) fn movable_context<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: LocalDefId,
    body: &mir::Body<'tcx>,
    part: &SharedPart,
    run: &Run<'tcx>,
) -> Option<MoveContext> {
    struct ItemStatement;
    impl<'tcx> Visitor<'tcx> for ItemStatement {
        type Result = ControlFlow<()>;
        fn visit_stmt(&mut self, stmt: &'tcx Stmt<'tcx>) -> ControlFlow<()> {
            if let StmtKind::Item(_) = stmt.kind {
                return ControlFlow::Break(());
            }
            intravisit::walk_stmt(self, stmt)
        }
    }

    let hir_id = tcx.local_def_id_to_hir_id(def);
    let owner = tcx.hir_node(hir_id);
    if owner.fn_sig()?.header.is_unsafe()
        || ItemStatement
            .visit_body(tcx.hir_body_owned_by(def))
            .is_break()
        || clippy_utils::inherits_cfg(tcx, def)
    {
        return None;
    }
    let item = std::iter::once(owner)
        .chain(tcx.hir_parent_iter(hir_id).map(|(_, node)| node))
        .find_map(|node| {
            if let Node::Item(item) = node {
                Some(item)
            } else {
                None
            }
        })?;
    let placeable = matches!(
        item.kind,
        ItemKind::Fn { .. } | ItemKind::Impl(_) | ItemKind::Trait { .. }
    );
    let module = tcx.parent_module_from_def_id(def);
    let in_module = tcx.opt_local_parent(item.owner_id.def_id) == Some(module.to_local_def_id());
    if !placeable || !in_module || item.span.from_expansion() {
        return None;
    }
    // An attribute between the run's block and the item would not cover the new fn.
    let mut nodes = vec![item.hir_id()];
    nodes.extend(
        std::iter::once(run.block.hir_id)
            .chain(tcx.hir_parent_id_iter(run.block.hir_id))
            .take_while(|&id| id != item.hir_id()),
    );
    let attrs = || nodes.iter().flat_map(|&id| tcx.hir_attrs(id));
    let lint_level = attrs().any(|attr| rustc_lint::Level::from_opt_symbol(attr.name()).is_some());
    if lint_level || find_attr!(attrs(), CfgTrace(..) | CfgAttrTrace) {
        return None;
    }
    let name = tcx.opt_item_name(def.to_def_id())?;
    let (module_items, _, _) = tcx.hir_get_module(module);
    let same_name = module_items
        .item_ids
        .iter()
        .flat_map(|&id| -> Vec<LocalDefId> {
            let item = tcx.hir_item(id);
            match item.kind {
                ItemKind::Fn { .. } => vec![item.owner_id.def_id],
                ItemKind::Impl(imp) => imp.items.iter().map(|i| i.owner_id.def_id).collect(),
                ItemKind::Trait { items, .. } => items.iter().map(|i| i.owner_id.def_id).collect(),
                _ => Vec::new(),
            }
        })
        .any(|other| {
            other != def
                && tcx.opt_item_name(other.to_def_id()) == Some(name)
                && tcx.generics_of(other).requires_monomorphization(tcx)
        });
    if same_name {
        return None;
    }
    let of_trait = match item.kind {
        ItemKind::Impl(imp) => match imp.of_trait {
            Some(header) => Some(header.trait_ref.trait_def_id()?),
            None => None,
        },
        _ => None,
    };
    let free = run_is_movable(tcx, def, of_trait, run)?;
    // A local the run names but does not bind (`let _ = x;` is no MIR read) must be a parameter.
    let params: Vec<(Symbol, Span)> = part
        .params
        .iter()
        .filter_map(|&local| user_var(body, local))
        .collect();
    if free
        .iter()
        .any(|&id| !params.contains(&(tcx.hir_name(id), tcx.hir_span(id))))
    {
        return None;
    }
    Some(MoveContext {
        insert_at: item.span.shrink_to_hi(),
        const_fn: tcx.is_const_fn(def),
        module,
    })
}
