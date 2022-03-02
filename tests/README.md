# pySAR Tests <a name="TOP"></a>

All of the modules and functionalities of pySAR are thoroughly tested using the Python [unittest][unittest] framework.
## Module tests:

* `test_aaindex` - tests for AAIndex class.
* `test_descriptors` - tests for descriptors class.
* `test_encoding` - tests for encoding class.
* `test_evaluate` - tests for evaluate class.
* `test_model` - tests for model class.
* `test_pyDSP` - tests for pyDSP class.
* `test_pySAR` - tests for pySAR class.
* `test_utils` - tests for utils class.
* `test_get_protein` - tests for all functions in get_protein module.

## Running Tests

To run all unittests, make sure you are in the main pySAR directory and from a terminal/cmd-line run:
```python
python -m unittest discover tests -v

#-v produces a more verbose and useful output
```

To run a module's specific unittests, make sure you are in the pySAR directory and from a terminal/cmd-line run:
```python
python -m unittest tests.test_MODULE -v

```

To run a specific test case within one of the module's specific unittests. For example, testing the dipeptide composition descriptor functionality within the test_descriptors test suite. Firstly, make sure you are in the pySAR directory and from a terminal/cmd-line run:
```python
python -m unittest tests.test_descriptors.test_dipeptide_composition -v
```

## Directory folders:

* `/test_data` - stores all required test data and datasets used to test pySAR's functionality.
* `/test_config` - stores all required test configuration files used to test pySAR's functionality.

[unittest]: https://docs.python.org/3/library/unittest.html


