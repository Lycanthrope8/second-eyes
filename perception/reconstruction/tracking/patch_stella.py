"""Remove stella_vslam's dependency on g2o's LGPL CSparse solver (D132).

stella 0.7.0 uses g2o's CSparse linear solver in two places (global bundle adjustment, loop-closure pose graph) and
g2o's Eigen linear solver in three others. This patch makes the two use the Eigen solver as well, and stops linking
the CSparse components. The normal equations and their solution are unchanged; only the sparse factorisation library
differs. With Eigen 3.4.0, whose sparse ordering is MPL-2.0, no LGPL code remains in the build.
Every replacement is checked: if the sources differ from 0.7.0, the patch stops instead of guessing."""
import sys
from pathlib import Path

EDITS = {
    "src/stella_vslam/optimize/global_bundle_adjuster.cc": [
        ("#include <g2o/solvers/csparse/linear_solver_csparse.h>\n", ""),
        ("g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>", "g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>"),
    ],
    "src/stella_vslam/optimize/graph_optimizer.cc": [
        ("#include <g2o/solvers/csparse/linear_solver_csparse.h>", "#include <g2o/solvers/eigen/linear_solver_eigen.h>"),
        ("g2o::LinearSolverCSparse<g2o::BlockSolver_7_3::PoseMatrixType>", "g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>"),
    ],
    "src/stella_vslam/CMakeLists.txt": [
        ("             g2o::solver_eigen\n             g2o::solver_csparse\n             g2o::csparse_extension\n             OPTIONAL_COMPONENTS\n             g2o::csparse)",
         "             g2o::solver_eigen)"),
        ("                      g2o::solver_eigen\n                      g2o::solver_csparse\n                      g2o::csparse_extension\n",
         "                      g2o::solver_eigen\n"),
    ],
}
MARK = ".qrecon_patched"


def apply(root):
    root = Path(root)
    if (root / MARK).exists():
        return "already patched"
    for rel, edits in EDITS.items():
        p = root / rel
        s = p.read_text(encoding="utf-8")
        for old, new in edits:
            if s.count(old) != 1:
                raise SystemExit(f"patch_stella: expected exactly one match in {rel} for:\n{old[:120]}\n"
                                 "The sources are not stella_vslam 0.7.0 as pinned; stopping.")
            s = s.replace(old, new)
        p.write_text(s, encoding="utf-8")
    left = [str(p) for p in (root / "src").rglob("*") if p.is_file() and p.suffix in (".cc", ".h", ".txt")
            and ("LinearSolverCSparse" in p.read_text(encoding="utf-8", errors="ignore")
                 or "csparse_extension" in p.read_text(encoding="utf-8", errors="ignore"))]
    if left:
        raise SystemExit(f"patch_stella: CSparse still referenced in {left}")
    (root / MARK).write_text("CSparse solver replaced by g2o's Eigen solver (D132)\n", encoding="utf-8")
    return "patched"


if __name__ == "__main__":
    print(apply(sys.argv[1]))
