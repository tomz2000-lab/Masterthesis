Docstring Guidelines
====================

This document shows the recommended docstring formats for the Negotiation Platform.
The documentation system supports both Google-style and NumPy-style docstrings.

Google-Style Docstrings (Recommended)
--------------------------------------

Use this format for new functions and methods:

.. code-block:: python

    def example_function(param1, param2, optional_param=None):
        """
        Brief description of the function.

        Longer description if needed. This can span multiple lines and include
        detailed explanations of the function's behavior, algorithms used, etc.

        Args:
            param1 (str): Description of the first parameter.
            param2 (int): Description of the second parameter.
            optional_param (float, optional): Description of optional parameter.
                Defaults to None.

        Returns:
            dict: Description of the return value. If returning a complex object,
                describe its structure:
                
                - key1 (str): Description of key1
                - key2 (list): Description of key2

        Raises:
            ValueError: Description of when this exception is raised.
            FileNotFoundError: Description of when this exception is raised.

        Example:
            Basic usage example:

            >>> result = example_function("test", 42)
            >>> print(result)
            {'status': 'success', 'data': [1, 2, 3]}

            More complex example:

            >>> result = example_function("advanced", 100, optional_param=0.5)
            >>> print(result['status'])
            'success'

        Note:
            Any important notes about the function, limitations, or
            special considerations.

        See Also:
            related_function: Description of relationship.
            AnotherClass.method: Cross-reference to related methods.
        """

NumPy-Style Docstrings (Also Supported)
----------------------------------------

This format is already used in some of your code:

.. code-block:: python

    def numpy_style_function(param1, param2):
        """
        Brief description of the function.

        Longer description if needed.

        Parameters
        ----------
        param1 : str
            Description of the first parameter.
        param2 : int
            Description of the second parameter.

        Returns
        -------
        dict
            Description of the return value.

        Raises
        ------
        ValueError
            Description of when this exception is raised.

        Examples
        --------
        >>> result = numpy_style_function("test", 42)
        >>> print(result)
        {'status': 'success'}
        """

Class Docstrings
-----------------

For classes, include an overview and document the __init__ method:

.. code-block:: python

    class ExampleClass:
        """
        Brief description of the class.

        Longer description explaining the purpose, main functionality,
        and how to use the class.

        Args:
            init_param (str): Description of initialization parameter.

        Attributes:
            public_attribute (int): Description of public attribute.
            _private_attribute (str): Description of private attribute.

        Example:
            >>> obj = ExampleClass("initialization_value")
            >>> result = obj.main_method()
            >>> print(result)
            'expected_output'
        """

        def __init__(self, init_param):
            """Initialize the ExampleClass.
            
            Args:
                init_param (str): Parameter for initialization.
            """

        def main_method(self):
            """
            Main method description.

            Returns:
                str: Description of return value.
            """

Module-Level Docstrings
-----------------------

Each Python file should start with a module docstring:

.. code-block:: python

    """
    Module Name
    ===========

    Brief description of what this module does.

    This module contains classes and functions for [specific purpose].
    Key functionality includes:

    - Feature 1: Brief description
    - Feature 2: Brief description
    - Feature 3: Brief description

    Example:
        Basic usage of the module:

        >>> from module_name import main_function
        >>> result = main_function()

    Note:
        Any important notes about the module, dependencies, or usage.
    """

Best Practices
--------------

1. **Be Consistent**: Choose either Google or NumPy style and stick to it within a file
2. **Be Descriptive**: Explain not just what, but why and how
3. **Include Examples**: Real usage examples help users understand
4. **Document Exceptions**: List all exceptions that can be raised
5. **Type Hints**: Include type information in docstrings or use Python type hints
6. **Cross-References**: Link to related functions/classes using Sphinx syntax

Type Hints Integration
----------------------

Combine docstrings with Python type hints for the best documentation:

.. code-block:: python

    from typing import Dict, List, Optional, Union

    def typed_function(
        data: List[str], 
        config: Dict[str, Union[str, int]], 
        optional_flag: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Function with type hints and docstring.

        Args:
            data: List of string data to process.
            config: Configuration dictionary with string or integer values.
            optional_flag: Optional boolean flag. Defaults to None.

        Returns:
            Dictionary containing processing results.
        """

Sphinx Integration
------------------

The documentation system will automatically:

- Extract all docstrings from your code
- Generate API documentation pages
- Create cross-references between modules
- Include examples in the documentation
- Generate search indices for all documented items

To ensure your docstrings appear in the documentation:

1. Add docstrings to all public functions, classes, and methods
2. Ensure your module has an __init__.py file (if it's a package)
3. The module is listed in docs/api/modules.rst or included via autosummary
4. Commit and push your changes - Read the Docs will rebuild automatically