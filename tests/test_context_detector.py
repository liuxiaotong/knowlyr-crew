"""测试项目类型检测器."""

import json

from crew.context_detector import ProjectInfo, detect_project


class TestProjectInfo:
    """测试 ProjectInfo 模型."""

    def test_display_label_unknown(self):
        info = ProjectInfo()
        assert info.display_label == "unknown"

    def test_display_label_python(self):
        info = ProjectInfo(project_type="python")
        assert info.display_label == "Python"

    def test_display_label_with_framework(self):
        info = ProjectInfo(project_type="python", framework="fastapi")
        assert info.display_label == "Python (Fastapi)"

    def test_display_label_nodejs(self):
        info = ProjectInfo(project_type="nodejs", framework="react")
        assert info.display_label == "Nodejs (React)"


class TestDetectProject:
    """测试 detect_project 函数."""

    def test_empty_dir_returns_unknown(self, tmp_path):
        info = detect_project(tmp_path)
        assert info.project_type == "unknown"

    def test_detect_python_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        info = detect_project(tmp_path)
        assert info.project_type == "python"
        assert "pyproject.toml" in info.key_files

    def test_detect_python_setup_py(self, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup\n")
        info = detect_project(tmp_path)
        assert info.project_type == "python"

    def test_detect_python_requirements(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask\n")
        info = detect_project(tmp_path)
        assert info.project_type == "python"

    def test_detect_python_uv(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        (tmp_path / "uv.lock").write_text("")
        info = detect_project(tmp_path)
        assert info.package_manager == "uv"

    def test_detect_python_poetry(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        (tmp_path / "poetry.lock").write_text("")
        info = detect_project(tmp_path)
        assert info.package_manager == "poetry"

    def test_detect_python_pytest(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
        info = detect_project(tmp_path)
        assert info.test_framework == "pytest"

    def test_detect_python_fastapi(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["fastapi"]\n')
        info = detect_project(tmp_path)
        assert info.framework == "fastapi"

    def test_detect_python_django(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["django"]\n')
        info = detect_project(tmp_path)
        assert info.framework == "django"

    def test_detect_python_ruff_lint(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 88\n")
        info = detect_project(tmp_path)
        assert "ruff" in info.lint_tools

    def test_detect_nodejs(self, tmp_path):
        pkg = {"name": "test", "dependencies": {"express": "^4.0"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        info = detect_project(tmp_path)
        assert info.project_type == "nodejs"
        assert info.framework == "express"
        assert info.package_manager == "npm"

    def test_detect_nodejs_react_jest(self, tmp_path):
        pkg = {
            "name": "test",
            "dependencies": {"react": "^18"},
            "devDependencies": {"jest": "^29"},
        }
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        info = detect_project(tmp_path)
        assert info.framework == "react"
        assert info.test_framework == "jest"

    def test_detect_nodejs_yarn(self, tmp_path):
        pkg = {"name": "test"}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        (tmp_path / "yarn.lock").write_text("")
        info = detect_project(tmp_path)
        assert info.package_manager == "yarn"

    def test_detect_nodejs_pnpm(self, tmp_path):
        pkg = {"name": "test"}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        (tmp_path / "pnpm-lock.yaml").write_text("")
        info = detect_project(tmp_path)
        assert info.package_manager == "pnpm"

    def test_detect_go(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/test\ngo 1.21\n")
        info = detect_project(tmp_path)
        assert info.project_type == "go"
        assert info.test_framework == "go test"

    def test_detect_rust(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'test'\n")
        info = detect_project(tmp_path)
        assert info.project_type == "rust"
        assert info.package_manager == "cargo"
        assert info.test_framework == "cargo test"

    def test_detect_java_maven(self, tmp_path):
        (tmp_path / "pom.xml").write_text("<project></project>")
        info = detect_project(tmp_path)
        assert info.project_type == "java"
        assert info.package_manager == "maven"

    def test_detect_java_gradle(self, tmp_path):
        (tmp_path / "build.gradle").write_text("apply plugin: 'java'")
        info = detect_project(tmp_path)
        assert info.project_type == "java"
        assert info.package_manager == "gradle"

    def test_detect_lint_eslintrc(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name":"x"}')
        (tmp_path / ".eslintrc.json").write_text("{}")
        info = detect_project(tmp_path)
        assert "eslint" in info.lint_tools

    def test_detect_editorconfig(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / ".editorconfig").write_text("[*]\nindent_style = space\n")
        info = detect_project(tmp_path)
        assert ".editorconfig" in info.key_files

    def test_python_priority_over_nodejs(self, tmp_path):
        """当同时存在 pyproject.toml 和 package.json 时，Python 优先."""
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / "package.json").write_text('{"name":"x"}')
        info = detect_project(tmp_path)
        assert info.project_type == "python"


class TestCacheThreadSafety:
    """缓存线程安全测试."""

    def test_concurrent_detect(self, tmp_path):
        """多线程并发检测不应崩溃."""
        import threading

        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
        errors = []

        def _detect():
            try:
                for _ in range(20):
                    info = detect_project(tmp_path, cache_ttl=0.01)
                    assert info.project_type == "python"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_detect) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_cache_ttl_zero_no_cache(self, tmp_path):
        """cache_ttl=0 应每次重新检测."""
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        info1 = detect_project(tmp_path, cache_ttl=0)
        assert info1.project_type == "python"
        # 修改后重新检测应反映变化
        (tmp_path / "pyproject.toml").unlink()
        (tmp_path / "package.json").write_text('{"name":"x"}')
        info2 = detect_project(tmp_path, cache_ttl=0)
        assert info2.project_type == "nodejs"
