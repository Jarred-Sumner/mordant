//! Prints a type as source text, for the new fn's signature.

use rustc_abi::ExternAbi;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalModDefId;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, GenericArgKind, Region, Ty, TyCtxt};
use rustc_span::sym;

/// `ty` as source text valid in `home`, or `None`. Raw pointers are allowed only `as_input`.
pub(super) fn printed_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    home: LocalModDefId,
    ty: Ty<'tcx>,
    as_input: bool,
) -> Option<String> {
    let list = |tys: &[Ty<'tcx>], as_input: bool| -> Option<Vec<String>> {
        tys.iter()
            .map(|&ty| printed_ty(tcx, home, ty, as_input))
            .collect()
    };
    let lifetime = |region: ty::Region<'tcx>| -> Option<&'static str> {
        match region.kind() {
            ty::ReStatic => Some("'static"),
            ty::ReErased if as_input => Some(""),
            _ if as_input && anonymous(tcx, region) => Some(""),
            _ => None,
        }
    };
    Some(match *ty.kind() {
        ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Str => ty.to_string(),
        ty::Ref(region, inner, mutbl) if as_input => {
            let lifetime = match lifetime(region)? {
                "" => String::new(),
                named => format!("{named} "),
            };
            format!(
                "&{lifetime}{}{}",
                mutbl.prefix_str(),
                printed_ty(tcx, home, inner, true)?
            )
        }
        ty::RawPtr(inner, mutbl) if as_input => {
            format!(
                "*{} {}",
                mutbl.ptr_str(),
                printed_ty(tcx, home, inner, true)?
            )
        }
        ty::Array(elem, len) => format!(
            "[{}; {}]",
            printed_ty(tcx, home, elem, as_input)?,
            len.try_to_target_usize(tcx)?
        ),
        ty::Slice(elem) => format!("[{}]", printed_ty(tcx, home, elem, as_input)?),
        ty::Tuple(tys) => match &list(tys, as_input)?[..] {
            [one] => format!("({one},)"),
            all => format!("({})", all.join(", ")),
        },
        ty::FnPtr(sig_tys, header) => {
            let sig = sig_tys.no_bound_vars()?;
            if header.c_variadic() {
                return None;
            }
            let abi = match header.abi() {
                ExternAbi::Rust => String::new(),
                abi => format!("extern {abi} "),
            };
            let inputs = list(sig.inputs(), false)?.join(", ");
            let output = match sig.output() {
                unit if unit.is_unit() => String::new(),
                ty => format!(" -> {}", printed_ty(tcx, home, ty, false)?),
            };
            format!("{}{abi}fn({inputs}){output}", header.safety().prefix_str())
        }
        ty::Adt(adt, args) => {
            let did = adt.did();
            let path = if let Some(local) = did.as_local() {
                // Nameable from `home`: every parent is a module and the nearest contains it.
                let parent = tcx.opt_local_parent(local)?;
                let mut step = Some(parent);
                while let Some(at) = step {
                    if tcx.def_kind(at) != DefKind::Mod {
                        return None;
                    }
                    step = tcx.opt_local_parent(at);
                }
                if !tcx.is_descendant_of(home.to_def_id(), parent.to_def_id()) {
                    return None;
                }
                format!("crate::{}", with_no_trimmed_paths!(tcx.def_path_str(did)))
            } else {
                if !matches!(tcx.crate_name(did.krate), sym::core | sym::alloc | sym::std)
                    || !tcx.visible_parent_map(()).contains_key(&did)
                {
                    return None;
                }
                with_no_trimmed_paths!(tcx.def_path_str(did))
            };
            let own = tcx.generics_of(did).own_args_no_defaults(tcx, args);
            let printed = own
                .iter()
                .map(|arg| match arg.kind() {
                    GenericArgKind::Type(ty) => printed_ty(tcx, home, ty, as_input),
                    GenericArgKind::Lifetime(region) => match lifetime(region)? {
                        "" => Some("'_".to_owned()),
                        named => Some(named.to_owned()),
                    },
                    GenericArgKind::Const(_) => None,
                })
                .collect::<Option<Vec<_>>>()?;
            if printed.is_empty() {
                path
            } else {
                format!("{path}<{}>", printed.join(", "))
            }
        }
        _ => return None,
    })
}

/// A late-bound unnamed lifetime: one parameter's alone, so an elided one means the same.
pub(super) fn anonymous(tcx: TyCtxt<'_>, region: Region<'_>) -> bool {
    matches!(region.kind(), ty::ReLateParam(free)
        if free.kind.get_id().is_some() && !free.kind.is_named(tcx))
}
