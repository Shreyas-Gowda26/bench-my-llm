from src.bench_my_llm.runner import _count_tokens_approx
from src.bench_my_llm.prompts import Prompt, PromptSuite
from src.bench_my_llm.runner import run_single_prompt, run_benchmark, BenchmarkResult,BenchmarkRun
from unittest.mock import MagicMock, patch
from openai import AuthenticationError
import pytest
def test_count_tokens_approx_basic():
    # no mocking needed - just call it and assert!
    result = _count_tokens_approx("hello world")
    assert result == 2

def test_count_tokens_approx_empty():    
    result = _count_tokens_approx("")
    assert result == 1

def test_count_tokens_approx_long():    
    result = _count_tokens_approx("no mocking needed - just call it and assert!")
    assert result == 11

def test_run_single_prompt_basic():
    mock_client = MagicMock()
    
    # fake chunk that looks like a real OpenAI streaming chunk
    mock_chunk = MagicMock()
    mock_chunk.choices[0].delta.content = "Hello!"
    
    # create returns an iterable of chunks
    mock_client.chat.completions.create.return_value = [mock_chunk]
    
    prompt = Prompt(text="Say hello", category="test", max_tokens=10)
    result = run_single_prompt(mock_client, "gpt-4o", prompt)
    
    assert result.response_text == "Hello!"
    assert result.model == "gpt-4o"

def test_run_single_prompt_empty_response():
    mock_client = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.choices[0].delta.content = None  # no content
    mock_client.chat.completions.create.return_value = [mock_chunk]
    
    prompt = Prompt(text="Say hello", category="test", max_tokens=10)
    result = run_single_prompt(mock_client, "gpt-4o", prompt)
    
    assert result.response_text == ""  # empty response

def test_run_single_prompt_multiple_chunks():
    mock_client = MagicMock()
    chunk1 , chunk2,chunk3 = MagicMock(), MagicMock(), MagicMock()
    chunk1.choices[0].delta.content = "Hello"
    chunk2.choices[0].delta.content = " world"
    chunk3.choices[0].delta.content = "!"
    mock_client.chat.completions.create.return_value = [chunk1, chunk2, chunk3]
    prompt = Prompt(text="Say hello", category="test", max_tokens=10)
    result = run_single_prompt(mock_client, "gpt-4o", prompt)
    assert result.response_text == "Hello world!"


def test_run_benchmark_basic():
    with patch("src.bench_my_llm.runner.OpenAI") as mock_openai, \
         patch("src.bench_my_llm.runner.run_single_prompt") as mock_run:
        
        mock_run.return_value = BenchmarkResult(
            model="gpt-4o",
            prompt_text="Say hello",
            category="test",
            response_text="Hello!",
            ttft_ms=100.0,
            total_latency_ms=200.0,
            tokens_generated=5,
            tokens_per_second=25.0,
            prompt_tokens=3,
            completion_tokens=5,
        )
        
        suite = PromptSuite(name="test", description="test suite", prompts=[
        Prompt(text="Say hello", category="test", max_tokens=10)
        ])
        
        run = run_benchmark("gpt-4o", suite, api_key="fake-key")
        
        assert run.model == "gpt-4o"
        assert len(run.results) == 1

def test_run_benchmark_auth_error():
    with patch("src.bench_my_llm.runner.OpenAI"), \
         patch("src.bench_my_llm.runner.run_single_prompt") as mock_run:

        mock_run.side_effect = AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body={}
        )

        suite = PromptSuite(name="test", description="test suite", prompts=[
            Prompt(text="Say hello", category="test", max_tokens=10)
        ])

        with pytest.raises(SystemExit):
            run_benchmark("gpt-4o", suite, api_key="fake-key")


def test_run_benchmark_multiple_prompts():
    with patch("src.bench_my_llm.runner.OpenAI"), \
         patch("src.bench_my_llm.runner.run_single_prompt") as mock_run:

        mock_run.return_value = BenchmarkResult(
            model="gpt-4o",
            prompt_text="Say hello",
            category="test",
            response_text="Hello!",
            ttft_ms=100.0,
            total_latency_ms=200.0,
            tokens_generated=5,
            tokens_per_second=25.0,
            prompt_tokens=3,
            completion_tokens=5,
        )

        suite = PromptSuite(name="test", description="test suite", prompts=[
            Prompt(text="Prompt 1", category="test", max_tokens=10),
            Prompt(text="Prompt 2", category="test", max_tokens=10),
            Prompt(text="Prompt 3", category="test", max_tokens=10),
        ])

        run = run_benchmark("gpt-4o", suite, api_key="fake-key")
        assert len(run.results) == 3


def test_benchmark_run_save_and_load(tmp_path):
    run = BenchmarkRun(
        model="gpt-4o",
        suite_name="test",
        base_url="https://api.openai.com/v1",
        timestamp="2024-01-01T00:00:00+00:00",
    )
    run.results.append(BenchmarkResult(
        model="gpt-4o",
        prompt_text="Say hello",
        category="test",
        response_text="Hello!",
        ttft_ms=100.0,
        total_latency_ms=200.0,
        tokens_generated=5,
        tokens_per_second=25.0,
        prompt_tokens=3,
        completion_tokens=5,
    ))

    path = tmp_path / "results.json"
    run.save(path)
    loaded = BenchmarkRun.load(path)

    assert loaded.model == "gpt-4o"
    assert len(loaded.results) == 1
    assert loaded.results[0].response_text == "Hello!"