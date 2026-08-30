//! Finds one place in the source to point at for the chosen statements.

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::LocalDefId;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::{self, BasicBlock};
use rustc_middle::ty::TyCtxt;
use rustc_span::{BytePos, ExpnKind, Span};

use super::classify::{PlaceParams, counted_statement, counted_terminator};

/// Where in the source a part is, for a note. The finding's own span stays
/// the fn signature, which the baseline matches and an `#[allow]` goes on.
#[derive(Clone, Copy, Debug)]
pub(super) enum SourceSpan {
    Whole(Span),
    Start(Span),
}

/// Moves one MIR item's span to its innermost macro call site inside
/// `body_span`, in `body_span`'s context. A desugaring stays in place.
pub(super) fn body_position(mut span: Span, body_span: Span) -> Option<Span> {
    loop {
        let in_macro = span.from_expansion()
            && matches!(span.ctxt().outer_expn_data().kind, ExpnKind::Macro(..));
        if !in_macro && body_span.contains(span) {
            return Some(span.with_ctxt(body_span.ctxt()));
        }
        span = span.parent_callsite()?;
    }
}

/// The smallest span covering the part's counted items, after two corrections for MIR spans
/// that mislead (below). To see item spans, dump a fixture with `-Zdump-mir -Zmir-include-spans`.
pub(super) fn source_span<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: LocalDefId,
    body: &mir::Body<'tcx>,
    blocks: &DenseBitSet<BasicBlock>,
) -> Option<SourceSpan> {
    let body_span = tcx.hir_body_owned_by(def).value.span;
    let mut inside: Vec<Span> = Vec::new();
    let mut outside: Vec<(Span, bool)> = Vec::new();
    let mut places = PlaceParams {
        tcx,
        body,
        found: false,
    };
    for (block, data) in body.basic_blocks.iter_enumerated() {
        if data.is_cleanup {
            continue;
        }
        let in_part = blocks.contains(block);
        for statement in &data.statements {
            if counted_statement(statement)
                && let Some(span) = body_position(statement.source_info.span, body_span)
            {
                if in_part {
                    inside.push(span);
                } else {
                    outside.push((span, places.dependent_statement(statement)));
                }
            }
        }
        if let Some(terminator) = &data.terminator
            && counted_terminator(terminator)
            && let Some(span) = body_position(terminator.source_info.span, body_span)
        {
            if in_part {
                inside.push(span);
            } else {
                outside.push((span, places.dependent_terminator(terminator)));
            }
        }
    }
    let placed = inside.len();
    let by_position = |s: &Span| (s.lo(), std::cmp::Reverse(s.hi()));
    let first = *inside.iter().min_by_key(|s| by_position(s))?;

    // Equal spans: one expression compiled to both sides (a `for` head).
    let shared: FxHashSet<(BytePos, BytePos)> = inside.iter().map(|s| (s.lo(), s.hi())).collect();
    outside.retain(|(s, dependent)| *dependent || !shared.contains(&(s.lo(), s.hi())));

    // An item can span a whole construct (a block's `()`, a unit fn's `_0 = ()`, a read of a
    // call's result), so drop each item whose span contains an outside item's.
    outside.sort_by_key(|(s, _)| s.lo());
    let mut least_end = vec![BytePos(u32::MAX); outside.len() + 1];
    for i in (0..outside.len()).rev() {
        least_end[i] = least_end[i + 1].min(outside[i].0.hi());
    }
    let names_more = |item: &Span| {
        let from = outside.partition_point(|(s, _)| s.lo() < item.lo());
        least_end[from] <= item.hi()
    };
    inside.retain(|item| !names_more(item));
    inside.sort_by_key(by_position);
    let Some(&first_kept) = inside.first() else {
        return Some(SourceSpan::Start(first));
    };

    // Source order is not control-flow order: a `for` pattern is bound after the dependent
    // `next()`. So split at each dependent outside item and keep the segment with most items.
    let split_points: Vec<BytePos> = outside
        .iter()
        .filter(|(_, dependent)| *dependent)
        .map(|(s, _)| s.lo())
        .collect();
    let segment_of = |item: &Span| split_points.partition_point(|&at| at < item.lo());
    let mut segments: Vec<(usize, BytePos, BytePos)> = Vec::new();
    let mut open = None;
    for item in &inside {
        let segment = segment_of(item);
        match segments.last_mut() {
            Some((held, _, hi)) if open == Some(segment) => {
                *held += 1;
                *hi = (*hi).max(item.hi());
            }
            _ => {
                segments.push((1, item.lo(), item.hi()));
                open = Some(segment);
            }
        }
    }
    let (held, lo, hi) =
        segments
            .into_iter()
            .fold((0, first_kept.lo(), first_kept.hi()), |best, segment| {
                if segment.0 > best.0 { segment } else { best }
            });
    debug_assert!(
        !outside
            .iter()
            .any(|(s, dependent)| *dependent && lo <= s.lo() && s.hi() <= hi),
        "a dependent item inside the part's span"
    );
    Some(if held * 2 >= placed {
        SourceSpan::Whole(body_span.with_lo(lo).with_hi(hi))
    } else {
        SourceSpan::Start(first_kept)
    })
}
