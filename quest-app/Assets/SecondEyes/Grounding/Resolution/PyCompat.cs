using System;
using System.Collections.Generic;

namespace SecondEyes.Grounding.Resolution
{
    // A2.7 delivery 2 (D115): Python-exact arithmetic and ordering for the resolver's port, free of Unity. The PC
    // reference is CPython 3.12. Its min and max keep the first of equal values (0.0 and -0.0 compare equal); its sum
    // of floats starts from the int 0 and then uses Neumaier's compensated summation; its strings order by code point.
    // The port calls these helpers wherever the Python calls the builtins, so that every intermediate value, not only
    // every decision, matches the PC's.
    public static class Py
    {
        /// <summary>Python's min(a, b): the first unless the second is strictly smaller.</summary>
        public static double Min(double a, double b) { return b < a ? b : a; }

        /// <summary>Python's max(a, b): the first unless the second is strictly larger.</summary>
        public static double Max(double a, double b) { return b > a ? b : a; }

        /// <summary>Python's min over a non-empty sequence.</summary>
        public static double Min(IEnumerable<double> xs)
        {
            bool first = true;
            double cur = 0.0;
            foreach (double x in xs)
            {
                if (first) { cur = x; first = false; }
                else if (x < cur) cur = x;
            }
            if (first) throw new ArgumentException("min() of an empty sequence");
            return cur;
        }

        /// <summary>Python's max over a non-empty sequence.</summary>
        public static double Max(IEnumerable<double> xs)
        {
            bool first = true;
            double cur = 0.0;
            foreach (double x in xs)
            {
                if (first) { cur = x; first = false; }
                else if (x > cur) cur = x;
            }
            if (first) throw new ArgumentException("max() of an empty sequence");
            return cur;
        }

        /// <summary>CPython 3.12's sum() of floats: the int start 0 plus the first item (0 + -0.0 is 0.0), then
        /// Neumaier's compensated summation of the rest, the compensation added at the end when it is non-zero and
        /// finite. Python 3.11 and earlier add left to right; the PC reference runs 3.12.</summary>
        public static double Sum(IEnumerable<double> xs)
        {
            bool first = true;
            double f = 0.0, c = 0.0;
            foreach (double x in xs)
            {
                if (first) { f = 0.0 + x; first = false; continue; }
                double t = f + x;
                if (Math.Abs(f) >= Math.Abs(x)) c += (f - t) + x;
                else c += (x - t) + f;
                f = t;
            }
            if (first) return 0.0;
            if (c != 0.0 && !double.IsInfinity(c) && !double.IsNaN(c)) f += c;
            return f;
        }

        /// <summary>math.degrees: CPython multiplies by 180 / pi.</summary>
        public static double Degrees(double radians) { return RadToDeg * radians; }
        private static readonly double RadToDeg = 180.0 / Math.PI;

        /// <summary>Python's string order: by code point. The scene's identifiers are ASCII, where the ordinal
        /// comparison of UTF-16 code units is the same.</summary>
        public static int Compare(string a, string b) { return string.CompareOrdinal(a, b); }

        /// <summary>sorted() of strings.</summary>
        public static List<string> Sorted(IEnumerable<string> xs)
        {
            var l = new List<string>(xs);
            l.Sort(string.CompareOrdinal);
            return l;
        }

        /// <summary>dict.fromkeys(xs): the distinct items in first-seen order.</summary>
        public static List<string> Distinct(IEnumerable<string> xs)
        {
            var seen = new HashSet<string>();
            var l = new List<string>();
            foreach (string x in xs) if (seen.Add(x)) l.Add(x);
            return l;
        }
    }
}
