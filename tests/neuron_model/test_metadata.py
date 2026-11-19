

def test_labid_source_image_groups_fragments_of_one_snapshot():
    from src.neuron_model.metadata import labid_source_id, labid_unit_id

    a, b = labid_unit_id("Ab", "ab22-1_processed"), labid_unit_id("Ab", "ab22-2_processed")
    assert a != b and labid_source_id(a) == labid_source_id(b) == "Ab_22"
    assert labid_source_id(labid_unit_id("Ab+9009", "ab7-1")) == "Ab+9009_7"
    assert labid_source_id(labid_unit_id("Wt", "1003-22")) == "Wt_1003"
    assert labid_source_id("neuron_5") == "neuron_5"          # not a LabID id: its own unit


def test_split_by_source_image_keeps_fragments_together():
    import pandas as pd
    from src.neuron_model.spine_generation.data.splits import assert_no_leakage, build_group_split

    rows = [(f"labid/Ab_ab{n}_{k}/0/0/{i}", "labid", f"Ab_ab{n}_{k}", "disease", "none") for n in range(1, 31) for k in (1, 2) for i in range(3)]
    frame = pd.DataFrame(rows, columns=["spine_key", "dataset", "neuron_id", "health", "treatment"])
    splits = build_group_split(frame, group_level="source_image", stratify=["health", "treatment"], seed=1)
    assert_no_leakage(splits)
    assert splits["group_id"].nunique() == 30
    both = splits.assign(neuron=[k.split("/")[1] for k in splits["spine_key"]]).groupby(splits["group_id"])["split"].nunique()
    assert both.max() == 1


def test_labid_source_image_merges_wt_shared_stacks():
    from src.neuron_model.metadata import labid_source_id

    assert labid_source_id("Wt_4_1_processed") == labid_source_id("Wt_5_2") == "Wt_4+5"
    assert labid_source_id("Wt_8_1_processed_1") == labid_source_id("Wt_9_1") == "Wt_8+9"
    assert labid_source_id("Wt_6_1") == "Wt_6"
    assert labid_source_id("Ab_4_1") == "Ab_4"
    assert labid_source_id("Wt+9009_13_2") == "Wt+9009_13"
