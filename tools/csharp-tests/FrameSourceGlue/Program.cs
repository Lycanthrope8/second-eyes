using System;

/// <summary>
/// The real PassthroughFrameSource is Unity glue that no other check compiles. This project compiles it against stubs
/// of exactly the members it uses, so an error in it shows here before a Unity build. Building is the check.
/// </summary>
static class Program
{
    static int Main()
    {
        Console.WriteLine("  ok    the real PassthroughFrameSource compiles against the members it uses");
        Console.WriteLine("1 passed, 0 failed");
        return 0;
    }
}
