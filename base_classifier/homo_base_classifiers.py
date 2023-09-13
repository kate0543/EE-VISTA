from sklearn.ensemble import RandomForestClassifier


def get_homo(n_trees):
    prf1 = RandomForestClassifier(n_estimators=n_trees, max_features='sqrt', random_state=1, min_samples_split=0.1,
                                  n_jobs=1)

    prf2 = RandomForestClassifier(n_estimators=n_trees, max_features='sqrt', random_state=2, min_samples_split=0.1,
                                  n_jobs=1)

    crf1 = RandomForestClassifier(n_estimators=n_trees, max_features=1, random_state=1, min_samples_split=0.1, n_jobs=1)

    crf2 = RandomForestClassifier(n_estimators=n_trees, max_features=1, random_state=2, min_samples_split=0.1, n_jobs=1)

    return [prf1, prf2, crf1, crf2]
