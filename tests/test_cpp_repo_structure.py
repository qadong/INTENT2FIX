from INTENT2FIX.repo_structure import show_cpp_repo_structure


def test_cpp_repo_structure_keeps_navigation_files_and_filters_generated_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "format.cpp").write_text("void format() {}\n")
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "format.hpp").write_text("void format();\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("docs\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "generated.cpp").write_text("void generated() {}\n")
    (tmp_path / "CMakeLists.txt").write_text("project(example)\n")

    structure = show_cpp_repo_structure(str(tmp_path))

    assert structure == (
        "include/\n"
        "    format.hpp\n"
        "src/\n"
        "    format.cpp\n"
        "CMakeLists.txt"
    )
    assert "guide.md" not in structure
    assert "generated.cpp" not in structure
