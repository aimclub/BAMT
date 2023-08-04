Learning CompositeBN and sampling from it
=========================================

Here is a simple working example of how one can learn composite bn, look at the models
that were applied to nodes
and sample some data

.. code-block:: python
    data = pd.read_csv(r"data/benchmark/healthcare.csv", index_col=0)
    print(data.dtypes)
    encoder = preprocessing.LabelEncoder()
    p = pp.Preprocessor([("encoder", encoder)])

    preprocessed_data, _ = p.apply(data)
    print(preprocessed_data.head(5))
    bn = CompositeBN()

    info = p.info

    bn.add_nodes(info)

    bn.add_edges(data)

    bn.fit_parameters(data)
    bn.get_info(as_df=False)

    print(bn.sample(200))