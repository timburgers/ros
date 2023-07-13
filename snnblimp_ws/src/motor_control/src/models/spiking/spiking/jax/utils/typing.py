from typing import Any, Callable, Iterable, Tuple


__all__ = ["PRNGKey", "Shape", "Dtype", "Array", "InitFn", "SpikeFn"]

# often seen in flax code
# TODO: make these more specific?
PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any

# our own defs
InitFn = Callable[[PRNGKey, Shape, Dtype], Array]
SpikeFn = Callable[[Array], Tuple[Array, Callable[[Array], Array]]]
