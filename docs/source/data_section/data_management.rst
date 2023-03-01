Data Management
===============

BAMT is designed to hadle three types of data:

#. ``disc`` - discrete data (e.g. some sort of categorical data), python data types:
 ``['str', 'O', 'b', 'categorical', 'object', 'bool']``

#. ``disc_num`` - discrete numerical data, python data types: ``['int32', 'int64']``

#. ``cont`` - continuous data (e.g. some sort of numerical data), python data types: ``['float32', 'float64]``

If, for example, your data set contains a pd.Series of integers, but these integers are actually categories, you should
consider converting them to strings. This is because BAMT will treat them as ``disc_num`` data, which may not be what you want.
The same applies to floats, which will be treated as ``cont`` data.
