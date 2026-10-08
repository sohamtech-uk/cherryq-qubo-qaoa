from cherryq.lumi_sweep import SweepTask, build_task_matrix, shard_tasks


def test_lumi_task_matrix_and_sharding_cover_every_run_once():
    probabilities = {"current-cash": 0.25, "arrive-R1": 0.75}
    tasks = build_task_matrix(
        scenario_ids=("current-cash", "arrive-R1"),
        scenario_probabilities=probabilities,
        methods=("standard", "warm-start"),
        p_values=(1, 2),
        seeds=(11, 29, 47),
    )

    assert len(tasks) == 24
    assert all(isinstance(task, SweepTask) for task in tasks)

    shards = [shard_tasks(tasks, shard_index=i, shard_count=5) for i in range(5)]
    flattened = [task for shard in shards for task in shard]

    assert len(flattened) == len(tasks)
    assert set(flattened) == set(tasks)
