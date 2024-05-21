# This file is part of utils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np

__all__ = ("continueClass", "inClass", "TemplateMeta")


INTRINSIC_SPECIAL_ATTRIBUTES = frozenset(
    (
        "__qualname__",
        "__module__",
        "__metaclass__",
        "__dict__",
        "__weakref__",
        "__class__",
        "__subclasshook__",
        "__name__",
        "__doc__",
    )
)


def isAttributeSafeToTransfer(name: str, value: Any) -> bool:
    """Return True if an attribute is safe to monkeypatch-transfer to another
    class.

    This rejects special methods that are defined automatically for all
    classes, leaving only those explicitly defined in a class decorated by
    `continueClass` or registered with an instance of `TemplateMeta`.

    Parameters
    ----------
    name : `str`
        The name of the attribute to check.
    value : `~typing.Any`
        The value of the attribute.

    Returns
    -------
    `bool`
        Whether the attribute is safe to monkeypatch-transfer.
    """
    if name.startswith("__") and (
        value is getattr(object, name, None) or name in INTRINSIC_SPECIAL_ATTRIBUTES
    ):
        return False
    return True


def continueClass(cls):
    """Re-open the decorated class, adding any new definitions into the
    original.

    For example:

    .. code-block:: python

        class Foo:
            pass

        @continueClass
        class Foo:
            def run(self):
                return None

    is equivalent to:

    .. code-block:: python

        class Foo:
            def run(self):
                return None

    .. warning::

        Python's built-in `super` function does not behave properly in classes
        decorated with `continueClass`.  Base class methods must be invoked
        directly using their explicit types instead.
    """
    orig = getattr(sys.modules[cls.__module__], cls.__name__)
    for name in dir(cls):
        # Common descriptors like classmethod and staticmethod can only be
        # accessed without invoking their magic if we use __dict__; if we use
        # getattr on those we'll get e.g. a bound method instance on the dummy
        # class rather than a classmethod instance we can put on the target
        # class.
        attr = cls.__dict__.get(name, None) or getattr(cls, name)
        if isAttributeSafeToTransfer(name, attr):
            setattr(orig, name, attr)
    return orig


def inClass(cls, name: str | None = None):
    """Add the decorated function to the given class as a method.

    Parameters
    ----------
    name : `str` or `None`, optional
        Name to be associated with the decorated function if the default
        can not be determined.

    Examples
    --------
    For example:

    .. code-block:: python

        class Foo:
            pass

        @inClass(Foo)
        def run(self):
            return None

    is equivalent to:

    .. code-block:: python

        class Foo:
            def run(self):
                return None

    Notes
    -----
    Standard decorators like ``classmethod``, ``staticmethod``, and
    ``property`` may be used *after* this decorator.  Custom decorators
    may only be used if they return an object with a ``__name__`` attribute
    or the ``name`` optional argument is provided.
    """

    def decorate(func):
        # Using 'name' instead of 'name1' breaks the closure because
        # assignment signals a strictly local variable.
        name1 = name
        if name1 is None:
            if hasattr(func, "__name__"):
                name1 = func.__name__
            else:
                if hasattr(func, "__func__"):
                    # classmethod and staticmethod have __func__ but
                    # no __name__
                    name1 = func.__func__.__name__
                elif hasattr(func, "fget"):
                    # property has fget but no __name__
                    name1 = func.fget.__name__
                else:
                    raise ValueError(f"Could not guess attribute name for '{func}'.")
        setattr(cls, name1, func)
        return func

    return decorate


class TemplateMeta(type):
    """A metaclass for abstract base classes that tie together wrapped C++
    template types.

    C++ template classes are most easily wrapped with a separate Python class
    for each template type, which results in an unnatural Python interface.
    TemplateMeta provides a thin layer that connects these Python classes by
    giving them a common base class and acting as a factory to construct them
    in a consistent way.

    To use, simply create a new class with the name of the template class, and
    use ``TemplateMeta`` as its metaclass, and then call ``register`` on each
    of its subclasses.  This registers the class with a "type key" - usually a
    Python representation of the C++ template types.  The type key must be a
    hashable object - strings, type objects, and tuples of these (for C++
    classes with multiple template parameters) are good choices.  Alternate
    type keys for existing classes can be added by calling ``alias``, but only
    after a subclass already been registered with a "primary" type key.  For
    example::

    .. code-block:: python

        import numpy as np
        from ._image import ImageF, ImageD

        class Image(metaclass=TemplateMeta):
            pass

        Image.register(np.float32, ImageF)
        Image.register(np.float64, ImageD)
        Image.alias("F", ImageF)
        Image.alias("D", ImageD)

    We have intentionally used ``numpy`` types as the primary keys for these
    objects in this example, with strings as secondary aliases simply because
    the primary key is added as a ``dtype`` attribute on the the registered
    classes (so ``ImageF.dtype == numpy.float32`` in the above example).

    This allows user code to construct objects directly using ``Image``, as
    long as an extra ``dtype`` keyword argument is passed that matches one of
    the type keys::

    .. code-block:: python

        img = Image(52, 64, dtype=np.float32)

    This simply forwards additional positional and keyword arguments to the
    wrapped template class's constructor.

    The choice of "dtype" as the name of the template parameter is also
    configurable, and in fact multiple template parameters are also supported,
    by setting a ``TEMPLATE_PARAMS`` class attribute on the ABC to a tuple
    containing the names of the template parameters.  A ``TEMPLATE_DEFAULTS``
    attribute can also be defined to a tuple of the same length containing
    default values for the template parameters, allowing them to be omitted in
    constructor calls.  When the length of these attributes is more than one,
    the type keys passed to ``register`` and ``alias`` should be tuple of the
    same length; when the length of these attributes is one, type keys should
    generally not be tuples.

    As an aid for those writing the Python wrappers for C++ classes,
    ``TemplateMeta`` also provides a way to add pure-Python methods and other
    attributes to the wrapped template classes.  To add a ``sum`` method to
    all registered types, for example, we can just do::

    .. code-block:: python

        class Image(metaclass=TemplateMeta):

            def sum(self):
                return np.sum(self.getArray())

        Image.register(np.float32, ImageF)
        Image.register(np.float64, ImageD)

    .. note::

        ``TemplateMeta`` works by overriding the ``__instancecheck__`` and
        ``__subclasscheck__`` special methods, and hence does not appear in
        its registered subclasses' method resolution order or ``__bases__``
        attributes. That means its attributes are not inherited by registered
        subclasses. Instead, attributes added to an instance of
        ``TemplateMeta`` are *copied* into the types registered with it. These
        attributes will thus *replace* existing attributes in those classes
        with the same name, and subclasses cannot delegate to base class
        implementations of these methods.

    Finally, abstract base classes that use ``TemplateMeta`` define a dict-
    like interface for accessing their registered subclasses, providing
    something like the C++ syntax for templates::

    .. code-block:: python

        Image[np.float32] -> ImageF
        Image["D"] -> ImageD

    Both primary dtypes and aliases can be used as keys in this interface,
    which means types with aliases will be present multiple times in the dict.
    To obtain the sequence of unique subclasses, use the ``__subclasses__``
    method.

    .. warning::

        Python's built-in `super` function does not behave properly in classes
        that have `TemplateMeta` as their metaclass (which should be rare, as
        TemplateMeta ABCs will have base classes of their own)..
    """

    def __new__(cls, name, bases, attrs):
        # __new__ is invoked when the abstract base class is defined (via a
        # class statement).  We save a dict of class attributes (including
        # methods) that were defined in the class body so we can copy them
        # to registered subclasses later.
        # We also initialize an empty dict to store the registered subclasses.
        attrs["_inherited"] = {k: v for k, v in attrs.items() if isAttributeSafeToTransfer(k, v)}
        # The special "TEMPLATE_PARAMS" class attribute, if defined, contains
        # names of the template parameters, which we use to set those
        # attributes on registered subclasses and intercept arguments to the
        # constructor.  This line removes it from the dict of things that
        # should be inherited while setting a default of 'dtype' if it's not
        # defined.
        attrs["TEMPLATE_PARAMS"] = attrs["_inherited"].pop("TEMPLATE_PARAMS", ("dtype",))
        attrs["TEMPLATE_DEFAULTS"] = attrs["_inherited"].pop(
            "TEMPLATE_DEFAULTS", (None,) * len(attrs["TEMPLATE_PARAMS"])
        )
        attrs["_registry"] = {}
        self = type.__new__(cls, name, bases, attrs)

        if len(self.TEMPLATE_PARAMS) == 0:
            raise ValueError("TEMPLATE_PARAMS must be a tuple with at least one element.")
        if len(self.TEMPLATE_DEFAULTS) != len(self.TEMPLATE_PARAMS):
            raise ValueError("TEMPLATE_PARAMS and TEMPLATE_DEFAULTS must have same length.")
        return self

    def __call__(cls, *args, **kwds):
        # __call__ is invoked when someone tries to construct an instance of
        # the abstract base class.
        # If the ABC defines a "TEMPLATE_PARAMS" attribute, we use those
        # strings as the kwargs we should intercept to find the right type.
        # Generate a type mapping key from input keywords. If the type returned
        # from the keyword lookup is a numpy dtype object, fetch the underlying
        # type of the dtype
        key = []
        for p, d in zip(cls.TEMPLATE_PARAMS, cls.TEMPLATE_DEFAULTS):
            tempKey = kwds.pop(p, d)
            if isinstance(tempKey, np.dtype):
                tempKey = tempKey.type
            key.append(tempKey)
        key = tuple(key)

        # indices are only tuples if there are multiple elements
        clz = cls._registry.get(key[0] if len(key) == 1 else key, None)
        if clz is None:
            d = dict(zip(cls.TEMPLATE_PARAMS, key))
            raise TypeError(f"No registered subclass for {d}.")
        return clz(*args, **kwds)

    def __subclasscheck__(cls, subclass):
        # Special method hook for the issubclass built-in: we return true for
        # any registered type or true subclass thereof.
        if subclass in cls._registry.values():
            return True
        return any(issubclass(subclass, v) for v in cls._registry.values())

    def __instancecheck__(cls, instance):
        # Special method hook for the isinstance built-in: we return true for
        # an instance of any registered type or true subclass thereof.
        if type(instance) in cls._registry.values():
            return True
        return any(isinstance(instance, v) for v in cls._registry.values())

    def __subclasses__(cls):
        """Return a tuple of all classes that inherit from this class."""
        # This special method isn't defined as part of the Python data model,
        # but it exists on builtins (including ABCMeta), and it provides useful
        # functionality.
        return tuple(set(cls._registry.values()))

    def register(cls, key, subclass) -> None:
        """Register a subclass of this ABC with the given key (a string,
        number, type, or other hashable).

        Register may only be called once for a given key or a given subclass.

        Parameters
        ----------
        key : `str` or `numbers.Number` or `None` or `collections.abc.Hashable`
            Key to use for registration.
        subclass : `type`
            Subclass to register.
        """
        if key is None:
            raise ValueError("None may not be used as a key.")
        if subclass in cls._registry.values():
            raise ValueError("This subclass has already registered with another key; use alias() instead.")
        if cls._registry.setdefault(key, subclass) != subclass:
            if len(cls.TEMPLATE_PARAMS) == 1:
                d = {cls.TEMPLATE_PARAMS[0]: key}
            else:
                d = dict(zip(cls.TEMPLATE_PARAMS, key))
            raise KeyError(f"Another subclass is already registered with {d}")
        # If the key used to register a class matches the default key,
        # make the static methods available through the ABC
        if cls.TEMPLATE_DEFAULTS:
            defaults = cls.TEMPLATE_DEFAULTS[0] if len(cls.TEMPLATE_DEFAULTS) == 1 else cls.TEMPLATE_DEFAULTS
            if key == defaults:
                conflictStr = (
                    "Base class has attribute {}"
                    " which is a {} method of {}."
                    " Cannot link method to base class."
                )
                # In the following if statements, the explicit lookup in
                # __dict__ must be done, as a call to getattr returns the
                # bound method, which no longer reports as a static or class
                # method. The static methods must be transfered to the ABC
                # in this unbound state, so that python will still see them
                # as static methods and not attempt to pass self. The class
                # methods must be transfered to the ABC as a bound method
                # so that the correct cls be called with the class method
                for name in subclass.__dict__:
                    if name in ("__new__", "__init_subclass__"):
                        continue
                    obj = subclass.__dict__[name]
                    # copy over the static methods
                    isBuiltin = isinstance(obj, types.BuiltinFunctionType)
                    isStatic = callable(obj) and type(obj).__name__ == "nb_func" \
                        or isinstance(obj, staticmethod)
                    if isBuiltin or isStatic:
                        if hasattr(cls, name):
                            raise AttributeError(conflictStr.format(name, "static", subclass))
                        setattr(cls, name, obj)
                    # copy over the class methods
                    elif isinstance(obj, classmethod):
                        if hasattr(cls, name):
                            raise AttributeError(conflictStr.format(name, "class", subclass))
                        setattr(cls, name, getattr(subclass, name))

        def setattrSafe(name, value):
            try:
                currentValue = getattr(subclass, name)
                if currentValue != value:
                    msg = "subclass already has a '{}' attribute with value {} != {}."
                    raise ValueError(msg.format(name, currentValue, value))
            except AttributeError:
                setattr(subclass, name, value)

        if len(cls.TEMPLATE_PARAMS) == 1:
            setattrSafe(cls.TEMPLATE_PARAMS[0], key)
        elif len(cls.TEMPLATE_PARAMS) == len(key):
            for p, k in zip(cls.TEMPLATE_PARAMS, key):
                setattrSafe(p, k)
        else:
            raise ValueError(
                f"key must have {len(cls.TEMPLATE_PARAMS)} elements (one for each of {cls.TEMPLATE_PARAMS})"
            )

        for name, attr in cls._inherited.items():
            setattr(subclass, name, attr)

    def alias(cls, key, subclass) -> None:
        """Add an alias that allows an existing subclass to be accessed with a
        different key.

        Parameters
        ----------
        key : `str` or `numbers.Number` or `None` or `collections.abc.Hashable`
            Key to use for aliasing.
        subclass : `type`
            Subclass to alias.
        """
        if key is None:
            raise ValueError("None may not be used as a key.")
        if key in cls._registry:
            raise KeyError(f"Cannot multiply-register key {key}")
        primaryKey = tuple(getattr(subclass, p, None) for p in cls.TEMPLATE_PARAMS)
        if len(primaryKey) == 1:
            # indices are only tuples if there are multiple elements
            primaryKey = primaryKey[0]
        if cls._registry.get(primaryKey, None) != subclass:
            raise ValueError("Subclass is not registered with this base class.")
        cls._registry[key] = subclass

    # Immutable mapping interface defined below.  We don't use collections
    # mixins because we don't want their comparison operators.

    def __getitem__(cls, key):
        return cls._registry[key]

    def __iter__(cls):
        return iter(cls._registry)

    def __len__(cls):
        return len(cls._registry)

    def __contains__(cls, key):
        return key in cls._registry

    def keys(cls):
        """Return an iterable containing all keys (including aliases)."""
        return cls._registry.keys()

    def values(cls):
        """Return an iterable of registered subclasses, with duplicates
        corresponding to any aliases.
        """
        return cls._registry.values()

    def items(cls):
        """Return an iterable of (key, subclass) pairs."""
        return cls._registry.items()

    def get(cls, key, default=None) -> type:
        """Return the subclass associated with the given key.

        Parameters
        ----------
        key : `~collections.abc.Hashable`
            Key to query.
        default : `~typing.Any` or `None`, optional
            Default value to return if ``key`` is not found.

        Returns
        -------
        `type`
            Subclass with the given key. Includes aliases. Returns ``default``
            if the key is not recognized.
        """
        return cls._registry.get(key, default)
