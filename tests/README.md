# pySAR Tests <a name="TOP"></a>

All of the modules and functionalities of pySAR are thoroughly tested using the Python [unittest][unittest] framework.
## Module tests:

* `test_descriptors` - tests for descriptors module and class
* `test_encoding` - tests for encoding module and class.
* `test_model` - tests for model module and class.
* `test_pyDSP` - tests for pyDSP module and class.
* `test_pySAR` - tests for pySAR module and class.
* `test_utils` - tests for utils module and functionality
* `test_get_protein` - tests for all moule and functionality in get_protein module.

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

## Directory folders:

* `/test_data` - stores all required test data and datasets used to test pySAR's functionality.
* `/test_config` - stores all required test configuration files used to test pySAR's functionality.

[unittest]: https://docs.python.org/3/library/unittest.html


