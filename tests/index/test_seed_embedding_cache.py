from code_intent import seed_embedding_cache


def test_seed_records_skips_manifest_entry_without_vector_artifacts(tmp_path):
    assert (
        seed_embedding_cache._seed_records(
            index_dir=tmp_path,
            index_name="features",
            artifact_prefix="feature",
            cache_type="feature",
            repo_id="swe-bench_example__repo",
            builder="feature-intent-v5",
            model="voyage-2",
            dimension=1024,
            texts_for_metadata=lambda metadata: [],
        )
        == 0
    )
