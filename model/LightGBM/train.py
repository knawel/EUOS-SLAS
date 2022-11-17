

if __name__ == '__main__':
    # train model
    data_folder = config.config_data['dataset_filepath']
    X = pd.read_pickle(join(data_folder, "X.pk.zip"))
    y = pd.read_pickle(join(data_folder, "Y.pk.zip"))

    X_mat = X.values
    y_vec = y.values.flatten()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y_vec, test_size=0.12, random_state=123)
    sys.stdout.write(f"The number of features: {X_mat.shape[1]}\n")
    # collection of classifiers
    classifiers = config.classifiers

    preprocessor = StandardScaler()
    best_model = [None, 0]

    for cls in classifiers:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
            , ('classifier', classifiers[cls])
        ])
        model = pipeline.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = f1_score(predictions, y_test, average=None)
        score2 = cohen_kappa_score(predictions, y_test)
        print(f'Model:{cls}; score:{score2}; all_score:{score}')

        sum_score = score[0] + score[1]
        if score2 > best_model[1]:
            best_model = [model, score2]
    print(best_model)
    with open('model.pickle', 'wb') as handle:
        pickle.dump(best_model[0], handle, protocol=pickle.HIGHEST_PROTOCOL)





