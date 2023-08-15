Learning CompositeBN and sampling from it
=========================================

Here is a simple working example of how one can learn composite bn, look at the models
that were applied to nodes
and sample some data

.. code-block:: python

    # data reading and preprocessing
    data = pd.read_csv(r"data/benchmark/healthcare.csv", index_col=0)
    print(data.dtypes)
    encoder = preprocessing.LabelEncoder()
    p = pp.Preprocessor([("encoder", encoder)])

    preprocessed_data, _ = p.apply(data)
    print(preprocessed_data.head(5))

    # initialize empty network
    bn = CompositeBN()

    info = p.info

    # add initial nodes
    bn.add_nodes(info)

    # learn structure
    bn.add_edges(data)

    # learn parameters
    bn.fit_parameters(data)

    # get info about models in nodes
    bn.get_info(as_df=False)

    # sample some data
    data_sampled = bn.sample(200)

    print(data_sampled)