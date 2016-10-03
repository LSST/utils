#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

"""Support code for running unit tests"""
from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range

from contextlib import contextmanager
import gc
import inspect
import os
import subprocess
import sys
import unittest
import warnings
import numpy
import lsst.log

# File descriptor leak test will be skipped if psutil can not be imported
try:
    import psutil
except ImportError:
    psutil = None

try:
    import lsst.daf.base as dafBase
except ImportError:
    dafBase = None

try:
    type(memId0)
except NameError:
    memId0 = 0                          # ignore leaked blocks with IDs before memId0
    nleakPrintMax = 20                  # maximum number of leaked blocks to print

# Initialize the list of open files to an empty set
open_files = set()


def _get_open_files():
    """Return a set containing the list of open files."""
    if psutil is None:
        return set()
    return set(p.path for p in psutil.Process().open_files())


def init():
    """Initialize the memory tester and configure a default log."""
    global memId0
    global open_files
    if dafBase:
        memId0 = dafBase.Citizen_getNextMemId()  # used by MemoryTestCase
    # Reset the list of open files
    open_files = _get_open_files()
    # Setup a default configuration for log used in the test framework
    lsst.log.configure_prop("""
log4j.rootLogger=INFO, A1
log4j.appender.A1=ConsoleAppender
log4j.appender.A1.Target=System.err
log4j.appender.A1.layout=PatternLayout
log4j.appender.A1.layout.ConversionPattern=%-5p %c: %m%n
""")


def run(suite, exit=True):
    """!Exit with the status code resulting from running the provided test suite"""

    if unittest.TextTestRunner().run(suite).wasSuccessful():
        status = 0
    else:
        status = 1

    if exit:
        sys.exit(status)
    else:
        return status


def sort_tests(tests):
    """!Go through the supplied sequence of test suites and sort them to ensure that
    MemoryTestCases are at the end of the test list. Returns a combined
    TestSuite."""

    suite = unittest.TestSuite()
    memtests = []
    for test_suite in tests:
        try:
            # Just test the first test method in the suite for MemoryTestCase
            # Use loop rather than next as it is possible for a test class
            # to not have any test methods and the Python community prefers
            # for loops over catching a StopIteration exception.
            bases = None
            for method in test_suite:
                bases = inspect.getmro(method.__class__)
                break
            if bases is not None and MemoryTestCase in bases:
                memtests.append(test_suite)
            else:
                suite.addTests(test_suite)
        except TypeError:
            if isinstance(test_suite, MemoryTestCase):
                memtests.append(test_suite)
            else:
                suite.addTest(test_suite)
    suite.addTests(memtests)
    return suite


def suiteClassWrapper(tests):
    return unittest.TestSuite(sort_tests(tests))

# Replace the suiteClass callable in the defaultTestLoader
# so that we can reorder the test ordering. This will have
# no effect if no memory test cases are found.
unittest.defaultTestLoader.suiteClass = suiteClassWrapper

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class MemoryTestCase(unittest.TestCase):
    """!Check for memory leaks since memId0 was allocated"""

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        """!Reset the leak counter when the tests have been completed"""
        init()

    def testLeaks(self):
        """!Check for memory leaks in the preceding tests"""
        if dafBase:
            gc.collect()
            global memId0, nleakPrintMax
            nleak = dafBase.Citizen_census(0, memId0)
            if nleak != 0:
                plural = "s" if nleak != 1 else ""
                print("\n%d Object%s leaked:" % (nleak, plural))

                if nleak <= nleakPrintMax:
                    print(dafBase.Citizen_census(memId0))
                else:
                    census = dafBase.Citizen_census()
                    print("...")
                    for i in range(nleakPrintMax - 1, -1, -1):
                        print(census[i].repr())

                self.fail("Leaked %d block%s" % (nleak, plural))

    def testFileDescriptorLeaks(self):
        if psutil is None:
            self.skipTest("Unable to test file descriptor leaks. psutil unavailable.")
        gc.collect()
        global open_files
        now_open = _get_open_files()

        # Some files are opened out of the control of the stack.
        now_open = set(f for f in now_open if not f.endswith(".car") and not f.endswith(".ttf")
                                              and f != "/var/lib/sss/mc/passwd")

        diff = now_open.difference(open_files)
        if diff:
            for f in diff:
                print("File open: %s" % f)
            self.fail("Failed to close %d file%s" % (len(diff), "s" if len(diff) != 1 else ""))

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class ExecutablesTestCase(unittest.TestCase):
    """!Test that executables can be run and return good status.

    The test methods are dynamically created. Callers
    must subclass this class in their own test file and invoke
    the discover_tests() class method to register the tests.
    """
    TESTS_DISCOVERED = -1

    @classmethod
    def setUpClass(cls):
        """Abort testing if automated test creation was enabled and
        yet not tests were found."""

        if cls.TESTS_DISCOVERED == 0:
            raise Exception("No executables discovered.")

    def testSanity(self):
        """This test exists to ensure that there is at least one test to be
        executed. This allows the test runner to trigger the class set up
        machinery to test whether there are some executables to test."""
        pass

    def assertExecutable(self, executable, root_dir=None, args=None, msg=None):
        """!Check an executable runs and returns good status.

        @param executable: Path to an executable. root_dir is not used
        if this is an absolute path.

        @param root_dir: Directory containing exe. Ignored if None.

        @param args: List or tuple of arguments to be provided to the
        executable.

        @param msg: Message to use when the test fails. Can be None for
        default message.

        Prints output to standard out. On bad exit status the test
        fails. If the executable can not be located the test is skipped.
        """

        if root_dir is not None and not os.path.isabs(executable):
            executable = os.path.join(root_dir, executable)

        # Form the argument list for subprocess
        sp_args = [executable]
        argstr = "no arguments"
        if args is not None:
            sp_args.extend(args)
            argstr = 'arguments "' + " ".join(args) + '"'

        print("Running executable '{}' with {}...".format(executable, argstr))
        if not os.path.exists(executable):
            self.skipTest("Executable {} is unexpectedly missing".format(executable))
        failmsg = None
        try:
            output = subprocess.check_output(sp_args)
        except subprocess.CalledProcessError as e:
            output = e.output
            failmsg = "Bad exit status from '{}': {}".format(executable, e.returncode)
        print(output.decode('utf-8'))
        if failmsg:
            if msg is None:
                msg = failmsg
            self.fail(msg)

    @classmethod
    def _build_test_method(cls, executable, root_dir):
        """!Build a test method and attach to class.

        The method is built for the supplied excutable located
        in the supplied root directory.

        cls._build_test_method(root_dir, executable)

        @param cls The class in which to create the tests.

        @param executable Name of executable. Can be absolute path.

        @param root_dir Path to executable. Not used if executable path is absolute.
        """
        if not os.path.isabs(executable):
            executable = os.path.abspath(os.path.join(root_dir, executable))

        # Create the test name from the executable path.
        test_name = "test_exe_" + executable.replace("/", "_")

        # This is the function that will become the test method
        def test_executable_runs(*args):
            self = args[0]
            self.assertExecutable(executable)

        # Give it a name and attach it to the class
        test_executable_runs.__name__ = test_name
        setattr(cls, test_name, test_executable_runs)

    @classmethod
    def create_executable_tests(cls, ref_file, executables=None):
        """!Discover executables to test and create corresponding test methods.

        Scans the directory containing the supplied reference file
        (usually __file__ supplied from the test class) and look for
        executables. If executables are found a test method is created
        for each one. That test method will run the executable and
        check the returned value.

        Executable scripts with a .py extension and shared libraries
        are ignored by the scanner.

        This class method must be called before test discovery.

        cls.discover_tests(__file__)

        The list of executables can be overridden by passing in a
        sequence of explicit executables that should be tested.
        If an item in the sequence can not be found the
        test will be configured to skip rather than fail.
        """

        # Get the search directory from the reference file
        ref_dir = os.path.abspath(os.path.dirname(ref_file))

        if executables is None:
            # Look for executables to test by walking the tree
            executables = []
            for root, dirs, files in os.walk(ref_dir):
                for f in files:
                    # Skip Python files. Shared libraries are exectuable.
                    if not f.endswith(".py") and not f.endswith(".so"):
                        full_path = os.path.join(root, f)
                        if os.access(full_path, os.X_OK):
                            executables.append(full_path)

        # Store the number of tests found for later assessment.
        # Do not raise an exception if we have no executables as this would
        # cause the testing to abort before the test runner could properly
        # integrate it into the failure report.
        cls.TESTS_DISCOVERED = len(executables)

        # Create the test functions and attach them to the class
        for e in executables:
            cls._build_test_method(e, ref_dir)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def findFileFromRoot(ifile):
    """!Find file which is specified as a path relative to the toplevel directory;
    we start in $cwd and walk up until we find the file (or throw IOError if it doesn't exist)

    This is useful for running tests that may be run from _dir_/tests or _dir_"""

    if os.path.isfile(ifile):
        return ifile

    ofile = None
    file = ifile
    while file != "":
        dirname, basename = os.path.split(file)
        if ofile:
            ofile = os.path.join(basename, ofile)
        else:
            ofile = basename

        if os.path.isfile(ofile):
            return ofile

        file = dirname

    raise IOError("Can't find %s" % ifile)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


@contextmanager
def getTempFilePath(ext):
    """!Return a path suitable for a temporary file and try to delete the file on success

    If the with block completes successfully then the file is deleted, if possible;
    failure results in a printed warning.
    If the block exits with an exception the file if left on disk so it can be examined.

    @param[in] ext  file name extension, e.g. ".fits"
    @return path for a temporary file. The path is a combination of the caller's file path
    and the name of the top-level function, as per this simple example:
    @code
    # file tests/testFoo.py
    import unittest
    import lsst.utils.tests
    class FooTestCase(unittest.TestCase):
        def testBasics(self):
            self.runTest()

        def runTest(self):
            with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
                # if tests/.tests exists then tmpFile = "tests/.tests/testFoo_testBasics.fits"
                # otherwise tmpFile = "testFoo_testBasics.fits"
                ...
                # at the end of this "with" block the path tmpFile will be deleted, but only if
                # the file exists and the "with" block terminated normally (rather than with an exception)
    ...
    @endcode
    """
    stack = inspect.stack()
    # get name of first function in the file
    for i in range(2, len(stack)):
        frameInfo = inspect.getframeinfo(stack[i][0])
        if i == 2:
            callerFilePath = frameInfo.filename
            callerFuncName = frameInfo.function
        elif callerFilePath == frameInfo.filename:
            # this function called the previous function
            callerFuncName = frameInfo.function
        else:
            break

    callerDir, callerFileNameWithExt = os.path.split(callerFilePath)
    callerFileName = os.path.splitext(callerFileNameWithExt)[0]
    outDir = os.path.join(callerDir, ".tests")
    if not os.path.isdir(outDir):
        outDir = ""
    outName = "%s_%s%s" % (callerFileName, callerFuncName, ext)
    outPath = os.path.join(outDir, outName)
    yield outPath
    if os.path.isfile(outPath):
        try:
            os.remove(outPath)
        except OSError as e:
            print("Warning: could not remove file %r: %s" % (outPath, e))
    else:
        print("Warning: could not find file %r" % (outPath,))

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class TestCase(unittest.TestCase):
    """!Subclass of unittest.TestCase that adds some custom assertions for
    convenience.
    """


def inTestCase(func):
    """!A decorator to add a free function to our custom TestCase class, while also
    making it available as a free function.
    """
    setattr(TestCase, func.__name__, func)
    return func

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


@inTestCase
def assertRaisesLsstCpp(testcase, excClass, callableObj, *args, **kwargs):
    warnings.warn("assertRaisesLsstCpp is deprecated; please just use TestCase.assertRaises",
                  DeprecationWarning)
    return testcase.assertRaises(excClass, callableObj, *args, **kwargs)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import functools


def debugger(*exceptions):
    """!Decorator to enter the debugger when there's an uncaught exception

    To use, just slap a "@debugger()" on your function.

    You may provide specific exception classes to catch as arguments to
    the decorator function, e.g., "@debugger(RuntimeError, NotImplementedError)".
    This defaults to just 'AssertionError', for use on unittest.TestCase methods.

    Code provided by "Rosh Oxymoron" on StackOverflow:
    http://stackoverflow.com/questions/4398967/python-unit-testing-automatically-running-the-debugger-when-a-test-fails
    """
    if not exceptions:
        exceptions = (AssertionError, )

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                import sys
                import pdb
                pdb.post_mortem(sys.exc_info()[2])
        return wrapper
    return decorator

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def plotImageDiff(lhs, rhs, bad=None, diff=None, plotFileName=None):
    """!Plot the comparison of two 2-d NumPy arrays.

    NOTE: this method uses matplotlib and imports it internally; it should be
    wrapped in a try/except block within packages that do not depend on
    matplotlib (including utils).

    @param[in]  lhs            LHS values to compare; a 2-d NumPy array
    @param[in]  rhs            RHS values to compare; a 2-d NumPy array
    @param[in]  bad            A 2-d boolean NumPy array of values to emphasize in the plots
    @param[in]  diff           difference array; a 2-d NumPy array, or None to show lhs-rhs
    @param[in]  plotFileName   Filename to save the plot to.  If None, the plot will be displayed in a
                               a window.
    """
    from matplotlib import pyplot
    if diff is None:
        diff = lhs - rhs
    pyplot.figure()
    if bad is not None:
        # make an rgba image that's red and transparent where not bad
        badImage = numpy.zeros(bad.shape + (4,), dtype=numpy.uint8)
        badImage[:, :, 0] = 255
        badImage[:, :, 1] = 0
        badImage[:, :, 2] = 0
        badImage[:, :, 3] = 255*bad
    vmin1 = numpy.minimum(numpy.min(lhs), numpy.min(rhs))
    vmax1 = numpy.maximum(numpy.max(lhs), numpy.max(rhs))
    vmin2 = numpy.min(diff)
    vmax2 = numpy.max(diff)
    for n, (image, title) in enumerate([(lhs, "lhs"), (rhs, "rhs"), (diff, "diff")]):
        pyplot.subplot(2, 3, n + 1)
        im1 = pyplot.imshow(image, cmap=pyplot.cm.gray, interpolation='nearest', origin='lower',
                            vmin=vmin1, vmax=vmax1)
        if bad is not None:
            pyplot.imshow(badImage, alpha=0.2, interpolation='nearest', origin='lower')
        pyplot.axis("off")
        pyplot.title(title)
        pyplot.subplot(2, 3, n + 4)
        im2 = pyplot.imshow(image, cmap=pyplot.cm.gray, interpolation='nearest', origin='lower',
                            vmin=vmin2, vmax=vmax2)
        if bad is not None:
            pyplot.imshow(badImage, alpha=0.2, interpolation='nearest', origin='lower')
        pyplot.axis("off")
        pyplot.title(title)
    pyplot.subplots_adjust(left=0.05, bottom=0.05, top=0.92, right=0.75, wspace=0.05, hspace=0.05)
    cax1 = pyplot.axes([0.8, 0.55, 0.05, 0.4])
    pyplot.colorbar(im1, cax=cax1)
    cax2 = pyplot.axes([0.8, 0.05, 0.05, 0.4])
    pyplot.colorbar(im2, cax=cax2)
    if plotFileName:
        pyplot.savefig(plotFileName)
    else:
        pyplot.show()


@inTestCase
def assertFloatsAlmostEqual(testCase, lhs, rhs, rtol=sys.float_info.epsilon,
                            atol=sys.float_info.epsilon, relTo=None,
                            printFailures=True, plotOnFailure=False,
                            plotFileName=None, invert=False, msg=None):
    """!Highly-configurable floating point comparisons for scalars and arrays.

    The test assertion will fail if all elements lhs and rhs are not equal to within the tolerances
    specified by rtol and atol.  More precisely, the comparison is:

    abs(lhs - rhs) <= relTo*rtol OR abs(lhs - rhs) <= atol

    If rtol or atol is None, that term in the comparison is not performed at all.

    When not specified, relTo is the elementwise maximum of the absolute values of lhs and rhs.  If
    set manually, it should usually be set to either lhs or rhs, or a scalar value typical of what
    is expected.

    @param[in]  testCase       unittest.TestCase instance the test is part of
    @param[in]  lhs            LHS value(s) to compare; may be a scalar or array-like of any dimension
    @param[in]  rhs            RHS value(s) to compare; may be a scalar or array-like of any dimension
    @param[in]  rtol           Relative tolerance for comparison; defaults to double-precision epsilon.
    @param[in]  atol           Absolute tolerance for comparison; defaults to double-precision epsilon.
    @param[in]  relTo          Value to which comparison with rtol is relative.
    @param[in]  printFailures  Upon failure, print all inequal elements as part of the message.
    @param[in]  plotOnFailure  Upon failure, plot the originals and their residual with matplotlib.
                               Only 2-d arrays are supported.
    @param[in]  plotFileName   Filename to save the plot to.  If None, the plot will be displayed in a
                               a window.
    @param[in]  invert         If True, invert the comparison and fail only if any elements *are* equal.
                               Used to implement assertFloatsNotEqual, which should generally be used instead
                               for clarity.
    @param[in] msg             String to append to the error message when assert fails.
    """
    if not numpy.isfinite(lhs).all():
        testCase.fail("Non-finite values in lhs")
    if not numpy.isfinite(rhs).all():
        testCase.fail("Non-finite values in rhs")
    diff = lhs - rhs
    absDiff = numpy.abs(lhs - rhs)
    if rtol is not None:
        if relTo is None:
            relTo = numpy.maximum(numpy.abs(lhs), numpy.abs(rhs))
        else:
            relTo = numpy.abs(relTo)
        bad = absDiff > rtol*relTo
        if atol is not None:
            bad = numpy.logical_and(bad, absDiff > atol)
    else:
        if atol is None:
            raise ValueError("rtol and atol cannot both be None")
        bad = absDiff > atol
    failed = numpy.any(bad)
    if invert:
        failed = not failed
        bad = numpy.logical_not(bad)
        cmpStr = "=="
        failStr = "are the same"
    else:
        cmpStr = "!="
        failStr = "differ"
    errMsg = []
    if failed:
        if numpy.isscalar(bad):
            if rtol is None:
                errMsg = ["%s %s %s; diff=%s with atol=%s"
                          % (lhs, cmpStr, rhs, absDiff, atol)]
            elif atol is None:
                errMsg = ["%s %s %s; diff=%s/%s=%s with rtol=%s"
                          % (lhs, cmpStr, rhs, absDiff, relTo, absDiff/relTo, rtol)]
            else:
                errMsg = ["%s %s %s; diff=%s/%s=%s with rtol=%s, atol=%s"
                          % (lhs, cmpStr, rhs, absDiff, relTo, absDiff/relTo, rtol, atol)]
        else:
            errMsg = ["%d/%d elements %s with rtol=%s, atol=%s"
                      % (bad.sum(), bad.size, failStr, rtol, atol)]
            if plotOnFailure:
                if len(lhs.shape) != 2 or len(rhs.shape) != 2:
                    raise ValueError("plotOnFailure is only valid for 2-d arrays")
                try:
                    plotImageDiff(lhs, rhs, bad, diff=diff, plotFileName=plotFileName)
                except ImportError:
                    errMsg.append("Failure plot requested but matplotlib could not be imported.")
            if printFailures:
                # Make sure everything is an array if any of them are, so we can treat
                # them the same (diff and absDiff are arrays if either rhs or lhs is),
                # and we don't get here if neither is.
                if numpy.isscalar(relTo):
                    relTo = numpy.ones(bad.shape, dtype=float) * relTo
                if numpy.isscalar(lhs):
                    lhs = numpy.ones(bad.shape, dtype=float) * lhs
                if numpy.isscalar(rhs):
                    rhs = numpy.ones(bad.shape, dtype=float) * rhs
                if rtol is None:
                    for a, b, diff in zip(lhs[bad], rhs[bad], absDiff[bad]):
                        errMsg.append("%s %s %s (diff=%s)" % (a, cmpStr, b, diff))
                else:
                    for a, b, diff, rel in zip(lhs[bad], rhs[bad], absDiff[bad], relTo[bad]):
                        errMsg.append("%s %s %s (diff=%s/%s=%s)" % (a, cmpStr, b, diff, rel, diff/rel))

    if msg is not None:
        errMsg.append(msg)
    testCase.assertFalse(failed, msg="\n".join(errMsg))


@inTestCase
def assertFloatsNotEqual(testCase, lhs, rhs, **kwds):
    """
    Fail a test if the given floating point values are equal to within the given tolerances.

    See assertClose for more information.
    """
    return assertFloatsAlmostEqual(testCase, lhs, rhs, invert=True, **kwds)


@inTestCase
def assertFloatsEqual(testCase, lhs, rhs, **kwargs):
    """
    Assert that lhs == rhs (both numeric types, whether scalar or array).

    See assertClose (called with rtol=atol=0) for more information.
    """
    return assertFloatsAlmostEqual(testCase, lhs, rhs, rtol=0, atol=0, **kwargs)


@inTestCase
def assertClose(*args, **kwargs):
    warnings.warn("assertClose is deprecated; please use TestCase.assertFloatsAlmostEqual",
                  DeprecationWarning)
    return assertFloatsAlmostEqual(*args, **kwargs)


@inTestCase
def assertNotClose(*args, **kwargs):
    warnings.warn("assertNotClose is deprecated; please use TestCase.assertFloatsNotEqual",
                  DeprecationWarning)
    return assertFloatsNotEqual(*args, **kwargs)
