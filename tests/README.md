# pySAR Tests <a name="TOP"></a>

All of the modules and functionalities of pySAR are thoroughly tested using the Python [unittest][unittest] framework. `pySAR` has hundreds of individual unit tests with 51 test functions and 6 test cases for each of the modules. Running all unit tests takes approximately X minutes.

Module Tests
------------
* `test_descriptors` - tests for descriptors module and class.
* `test_encoding` - tests for encoding module and class.
* `test_model` - tests for model module and class.
* `test_pyDSP` - tests for pyDSP module and class.
* `test_pySAR` - tests for pySAR module and class.
* `test_utils` - tests for utils module and functionality.

Running Tests
-------------
To run all unittests, make sure you are in the main pySAR directory and from a terminal/cmd-line run:
```python
python -m unittest discover tests -v

#-v produces a more verbose and useful output
```

To run a module's specific unittests, make sure you are in the `pySAR` directory and from a terminal/cmd-line run:
```python
python -m unittest tests.test_MODULE -b
#-b output during a passing test is discarded. Output is echoed normally on test fail or error and is added to the failure messages.
```

Directory Folders
-----------------
* `/test_data` - stores all required test data and datasets used to test pySAR's functionality.
* `/test_config` - stores all required test configuration files used to test pySAR's functionality.

[unittest]: https://docs.python.org/3/library/unittest.html