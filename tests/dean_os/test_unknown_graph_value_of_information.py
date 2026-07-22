from dean_os.unknown_graph import UnknownGraph, ValueOfInformationAssessment


def _validated(**changes):
    values = {
        "status": "validated",
        "uncertainty_type": "epistemic",
        "scenario_change_potential": 0.8,
        "confidence_change_potential": 0.6,
        "wrong_conclusion_blocking_value": 0.9,
        "decision_relevance": 0.8,
        "collection_feasibility": 0.7,
        "normalized_collection_cost": 0.3,
        "evidence_basis": ["linked hypothesis h1 changes if cancelled orders are confirmed"],
        "assessor": "reviewer_1",
        "assessed_at": "2026-07-12T00:00:00+00:00",
    }
    values.update(changes)
    return ValueOfInformationAssessment(**values)


def test_validated_assessment_calculates_ordinal_triage_score():
    assessment = _validated().calculate()
    assert assessment.triage_score is not None
    assert 0 < assessment.triage_score < 1


def test_draft_or_unattributed_assessment_cannot_score():
    draft = _validated(status="draft").calculate()
    unattributed = _validated(evidence_basis=[]).calculate()
    assert draft.triage_score is None
    assert unattributed.triage_score is None


def test_collector_backlog_places_validated_voi_before_legacy_priority():
    graph = UnknownGraph(domain="semiconductor")
    legacy = graph.add(
        "generic high priority gap", priority="critical", can_fix_with_collector=True
    )
    decision_relevant = graph.add(
        "cancelled orders", priority="medium", can_fix_with_collector=True,
        linked_hypothesis_ids=["h1"],
    )
    assert graph.assess_value_of_information(decision_relevant.id, _validated())

    ranked = graph.prioritized_collector_backlog()
    assert ranked[0].id == decision_relevant.id
    assert ranked[1].id == legacy.id
    assert ranked[1].voi.triage_score is None


def test_resolved_unknown_is_not_ranked():
    graph = UnknownGraph()
    entry = graph.add("resolved", can_fix_with_collector=True)
    graph.resolve(entry.id, "evidence obtained")
    assert graph.prioritized_collector_backlog() == []
