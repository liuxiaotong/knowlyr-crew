"""测试自定义异常体系."""

from crew.exceptions import (
    CrewError,
    EmployeeNotFoundError,
    EmployeeValidationError,
    ExecutorError,
    PipelineError,
    PipelineNotFoundError,
    ProviderError,
    TaskRegistryError,
)


class TestExceptionHierarchy:
    """异常继承关系."""

    def test_crew_error_is_exception(self):
        assert issubclass(CrewError, Exception)

    def test_all_inherit_crew_error(self):
        for cls in [
            EmployeeNotFoundError,
            EmployeeValidationError,
            PipelineNotFoundError,
            PipelineError,
            ExecutorError,
            ProviderError,
            TaskRegistryError,
        ]:
            assert issubclass(cls, CrewError), f"{cls.__name__} should inherit CrewError"

    def test_provider_error_is_value_error(self):
        """ProviderError 同时继承 ValueError 保持向后兼容."""
        assert issubclass(ProviderError, ValueError)
        # Should be catchable by except (ValueError, ImportError)
        try:
            raise ProviderError("test")
        except (ValueError, ImportError):
            pass  # Good, caught
        except Exception:
            assert False, "ProviderError should be caught by except ValueError"


class TestEmployeeNotFoundError:
    def test_message(self):
        err = EmployeeNotFoundError("code-reviewer")
        assert "code-reviewer" in str(err)
        assert err.name == "code-reviewer"

    def test_caught_by_crew_error(self):
        try:
            raise EmployeeNotFoundError("test")
        except CrewError:
            pass


class TestEmployeeValidationError:
    def test_message(self):
        err = EmployeeValidationError("bot", ["missing name", "empty body"])
        assert "bot" in str(err)
        assert "missing name" in str(err)
        assert err.name == "bot"
        assert len(err.errors) == 2


class TestPipelineNotFoundError:
    def test_message(self):
        err = PipelineNotFoundError("full-review")
        assert "full-review" in str(err)
        assert err.name == "full-review"


class TestProviderError:
    def test_message(self):
        err = ProviderError("无法识别模型 'foo'")
        assert "foo" in str(err)

    def test_is_both_crew_and_value_error(self):
        err = ProviderError("test")
        assert isinstance(err, CrewError)
        assert isinstance(err, ValueError)
