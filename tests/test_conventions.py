"""Hold every suite and every algorithm to the shape the platform expects.

These are the conventions the rest of the platform reads: the GUI builds its
forms from constructor signatures, the compatibility gate reads
algorithm_information, and the analysis tools read SETTINGS. A module that
drifts from them still imports, so nothing catches it until something downstream
quietly misbehaves.
"""
import importlib
import inspect
import pkgutil

import pytest

from ddmtolab.Methods.mtop import MTOP

PROBLEM_CATEGORIES = ('STSO', 'STMO', 'MTSO', 'MTMO', 'RWO')
ALGORITHM_CATEGORIES = ('STSO', 'STMO', 'MTSO', 'MTMO')

PROBLEM_INFO_KEYS = {'n_cases', 'n_cons', 'n_dims', 'n_objs', 'n_tasks', 'type'}
ALGORITHM_INFO_KEYS = {'cons', 'dims', 'expensive', 'knowledge_transfer',
                       'max_nfes', 'n_cons', 'n_objs', 'n_tasks', 'objs'}
SHARED_ALGORITHM_PARAMS = ('problem', 'max_nfes', 'save_data', 'save_path', 'name')


def _returns_mtop(f):
    annotation = inspect.signature(f).return_annotation
    return annotation is MTOP or (isinstance(annotation, str)
                                  and annotation.strip("'\"") == 'MTOP')


def problem_suites():
    """Every (category, module, class) that exposes benchmark problems."""
    found = []
    for cat in PROBLEM_CATEGORIES:
        pkg = importlib.import_module(f'ddmtolab.Problems.{cat}')
        for m in sorted(x.name for x in pkgutil.iter_modules(pkg.__path__)):
            mod = importlib.import_module(f'ddmtolab.Problems.{cat}.{m}')
            for name, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mod.__name__:
                    continue
                methods = [n for n, f in inspect.getmembers(cls, inspect.isfunction)
                           if not n.startswith('_') and _returns_mtop(f)]
                if methods:
                    found.append((f'{cat}/{m}', mod, cls, sorted(methods)))
    return found


def algorithms():
    """Every (label, module, class) under Algorithms."""
    found = []
    for cat in ALGORITHM_CATEGORIES:
        pkg = importlib.import_module(f'ddmtolab.Algorithms.{cat}')
        for m in sorted(x.name for x in pkgutil.iter_modules(pkg.__path__)):
            mod = importlib.import_module(f'ddmtolab.Algorithms.{cat}.{m}')
            found.append((f'{cat}/{m}', m, mod))
    return found


SUITES = problem_suites()
ALGOS = algorithms()


# ------------------------------------------------------------------ problems

def test_the_platform_ships_the_suites_the_docs_claim():
    assert len(SUITES) == 28
    assert sum(len(methods) for _, _, _, methods in SUITES) == 211


@pytest.mark.parametrize("label, mod, cls, methods", SUITES, ids=[s[0] for s in SUITES])
def test_a_suite_declares_itself_the_same_way_as_every_other(label, mod, cls, methods):
    info = getattr(cls, 'problem_information', None)
    assert info is not None, f'{label} has no problem_information'
    assert set(info) == PROBLEM_INFO_KEYS, f'{label} declares {sorted(info)}'
    assert info['n_cases'] == len(methods), \
        f'{label} declares {info["n_cases"]} cases but exposes {len(methods)}'


@pytest.mark.parametrize("label, mod, cls, methods", SUITES, ids=[s[0] for s in SUITES])
def test_a_suite_documents_itself(label, mod, cls, methods):
    assert (mod.__doc__ or '').strip(), f'{label} module has no docstring'
    assert (cls.__doc__ or '').strip(), f'{label} class has no docstring'
    for name in methods:
        f = getattr(cls, name)
        assert (f.__doc__ or '').strip(), f'{label}.{name} has no docstring'


@pytest.mark.parametrize("label, mod, cls, methods", SUITES, ids=[s[0] for s in SUITES])
def test_reference_fronts_accompany_exactly_the_multiobjective_suites(label, mod, cls, methods):
    """Read the declaration rather than build a problem: one PINN-HPO task
    trains a network before it can report how many objectives it has, and this
    also checks the declaration against what the module actually ships."""
    settings = getattr(mod, 'SETTINGS', None)
    multiobjective = str(cls.problem_information['n_objs']).strip() != '1'
    if multiobjective:
        assert settings is not None, f'{label} is multiobjective but ships no SETTINGS'
        assert 'metric' in settings, f'{label} SETTINGS names no metric'
        missing = [n for n in methods if n not in settings]
        assert not missing, f'{label} SETTINGS is missing {missing}'
    else:
        assert settings is None, f'{label} is single-objective but ships SETTINGS'


# ---------------------------------------------------------------- algorithms

def test_the_platform_ships_the_algorithms_the_docs_claim():
    assert len(ALGOS) == 115


@pytest.mark.parametrize("label, name, mod", ALGOS, ids=[a[0] for a in ALGOS])
def test_an_algorithm_is_named_after_its_module(label, name, mod):
    cls = getattr(mod, name, None)
    assert cls is not None, f'{label} has no class called {name}'
    assert (mod.__doc__ or '').strip(), f'{label} module has no docstring'
    assert (cls.__doc__ or '').strip(), f'{label} class has no docstring'


@pytest.mark.parametrize("label, name, mod", ALGOS, ids=[a[0] for a in ALGOS])
def test_an_algorithm_declares_itself_the_same_way_as_every_other(label, name, mod):
    cls = getattr(mod, name)
    info = getattr(cls, 'algorithm_information', None)
    assert info is not None, f'{label} has no algorithm_information'

    # Everything but the population-size key, which the next test pins.
    assert ALGORITHM_INFO_KEYS <= set(info), \
        f'{label} is missing {sorted(ALGORITHM_INFO_KEYS - set(info))}'
    assert isinstance(inspect.getattr_static(cls, 'get_algorithm_information'), classmethod), \
        f'{label}.get_algorithm_information is not a classmethod'
    assert (getattr(cls, 'optimize').__doc__ or '').strip(), \
        f'{label}.optimize has no docstring'


@pytest.mark.parametrize("label, name, mod", ALGOS, ids=[a[0] for a in ALGOS])
def test_the_size_parameter_says_whether_an_algorithm_is_expensive(label, name, mod):
    """An expensive algorithm sizes an initial design, a cheap one sizes a
    population, and both the declaration and the constructor have to say so."""
    cls = getattr(mod, name)
    info = cls.algorithm_information
    expensive = str(info.get('expensive')) == 'True'
    expected = 'n_initial' if expensive else 'n'
    other = 'n' if expensive else 'n_initial'

    assert expected in info, f'{label} is expensive={expensive} but declares no {expected!r}'
    assert other not in info, f'{label} is expensive={expensive} yet declares {other!r}'

    params = inspect.signature(cls.__init__).parameters
    assert expected in params, \
        f'{label} is expensive={expensive} but its constructor has no {expected!r}'


@pytest.mark.parametrize("label, name, mod", ALGOS, ids=[a[0] for a in ALGOS])
def test_every_algorithm_takes_the_shared_parameters(label, name, mod):
    params = list(inspect.signature(getattr(mod, name).__init__).parameters)
    assert params[1] == 'problem', f'{label} does not take problem first: {params[:3]}'
    for shared in SHARED_ALGORITHM_PARAMS:
        assert shared in params, f'{label} has no {shared!r} parameter'
