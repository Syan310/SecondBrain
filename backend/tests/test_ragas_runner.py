from evaluation.ragas_runner import (
    build_ragas_samples,
    build_ragas_judge_llm_kwargs,
    compute_ragas_means,
    strip_unsupported_sampling_args,
)


def test_build_ragas_samples_skips_missing_rag_results():
    qa_subset = [
        {"question": "q1", "ground_truth": "a1"},
        {"question": "q2", "ground_truth": "a2"},
    ]
    rag_results_map = {
        "q1": {
            "answer": "resp1",
            "context_texts": ["ctx1", "ctx2"],
        }
    }

    assert build_ragas_samples(qa_subset, rag_results_map) == [
        {
            "user_input": "q1",
            "response": "resp1",
            "retrieved_contexts": ["ctx1", "ctx2"],
            "reference": "a1",
        }
    ]


def test_compute_ragas_means_ignores_non_numeric_columns_and_nan():
    ragas_results = {
        "user_input": ["q1", "q2"],
        "response": ["a", "b"],
        "faithfulness": [0.5, 1.0],
        "answer_relevancy": [float("nan"), 0.8],
    }

    assert compute_ragas_means(ragas_results) == {
        "faithfulness": 0.75,
        "answer_relevancy": 0.8,
    }


def test_build_ragas_judge_llm_kwargs_only_sets_max_tokens():
    assert build_ragas_judge_llm_kwargs("anthropic/claude-sonnet-5.5") == {"max_tokens": 8192}


def test_strip_unsupported_sampling_args_removes_ragas_defaults():
    class FakeLLM:
        model_args = {"temperature": 0.01, "top_p": 0.1, "max_tokens": 8192}

    llm = strip_unsupported_sampling_args(FakeLLM())

    assert llm.model_args == {"max_tokens": 8192}
