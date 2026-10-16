using System;

namespace SecondEyes.Perception.Localization
{
    /// <summary>
    /// Minimum-cost assignment of rows to columns of a rectangular cost matrix (the Hungarian algorithm with potentials;
    /// A4 Delivery 1, docs/localization.md). An infinite cost is never assigned. The result depends only on the matrix.
    /// </summary>
    public static class Hungarian
    {
        /// <summary>For each row, its assigned column, or -1.</summary>
        public static int[] Solve(double[,] cost)
        {
            int rows = cost.GetLength(0), cols = cost.GetLength(1), n = Math.Max(rows, cols);
            var result = new int[rows];
            for (int i = 0; i < rows; i++) result[i] = -1;
            if (rows == 0 || cols == 0) return result;
            const double Big = 1e9;
            var a = new double[n + 1, n + 1];
            for (int i = 1; i <= n; i++)
                for (int j = 1; j <= n; j++)
                {
                    double c = i <= rows && j <= cols ? cost[i - 1, j - 1] : Big;
                    a[i, j] = double.IsNaN(c) || c > Big ? Big : c;
                }
            var u = new double[n + 1]; var v = new double[n + 1]; var p = new int[n + 1]; var way = new int[n + 1];
            for (int i = 1; i <= n; i++)
            {
                p[0] = i;
                int j0 = 0;
                var minv = new double[n + 1];
                var used = new bool[n + 1];
                for (int j = 0; j <= n; j++) minv[j] = double.PositiveInfinity;
                do
                {
                    used[j0] = true;
                    int i0 = p[j0], j1 = 0;
                    double delta = double.PositiveInfinity;
                    for (int j = 1; j <= n; j++)
                        if (!used[j])
                        {
                            double cur = a[i0, j] - u[i0] - v[j];
                            if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                            if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                        }
                    for (int j = 0; j <= n; j++)
                        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                        else minv[j] -= delta;
                    j0 = j1;
                } while (p[j0] != 0);
                do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0 != 0);
            }
            for (int j = 1; j <= n; j++)
            {
                int i = p[j];
                if (i >= 1 && i <= rows && j <= cols && a[i, j] < Big) result[i - 1] = j - 1;
            }
            return result;
        }
    }
}
