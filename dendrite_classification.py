import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns


scoring = {
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'roc_auc_ovr': make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
}

models = {
    'LogReg': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=0.01  
        ))
    ]),
    'LinSVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(
            kernel='linear',
            class_weight='balanced',
            probability=True,
            C=0.01  
        ))
    ]),
    'RandForest': RandomForestClassifier(
        n_estimators=50,
        max_depth=2,               
        min_samples_leaf=5,       
        min_samples_split=10,      
        class_weight='balanced',
        random_state=42
    ),
    'GBoost': GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.03,          
        max_depth=2,
        subsample=0.6,
        min_samples_split=15,
        min_samples_leaf=8,          
        random_state=42
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=2,
        num_leaves=15,             
        min_child_samples=10,
        class_weight='balanced',
        random_state=42
    )
}

def classification(path1, path2 = None):
    X_train, y_train, X_test, y_test = get_data(path1, path2)
    result_df = cross_validation(X_train, y_train)
    train_and_test(X_train, y_train, X_test, y_test, result_df)

def cross_validation(X_train, y_train):
    results = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, model in models.items():
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True
        )
        
        results.append({
            'model': name,
            'mean_train_f1': np.mean(cv_results['train_f1_weighted']),
            'mean_test_f1': np.mean(cv_results['test_f1_weighted']),
            'std_test_f1': np.std(cv_results['test_f1_weighted']),
            'mean_roc_auc_train': np.mean(cv_results['train_roc_auc_ovr']),
            'mean_roc_auc_test': np.mean(cv_results['test_roc_auc_ovr'])
        })

    # n_splits=3
    results_df = pd.DataFrame(results)
    # results_df.sort_values('mean_test_f1', ascending=False)
    sorted_df = results_df.sort_values('mean_test_f1', ascending=False)
    print(tabulate(sorted_df, headers='keys', tablefmt='psql', showindex=False))

    plt.figure(figsize=(10, 6))
    plt.bar(results_df['model'], results_df['mean_test_f1'], yerr=results_df['std_test_f1'], color = '#fd89c3')
    plt.title('Сравнение моделей по F1-score (weighted)', fontsize=18)
    plt.xlabel('Модель', fontsize=18)
    plt.ylabel('F1-score', fontsize=18)

    # Увеличение шрифта меток на осях
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 1)

    plt.savefig('f1.png', dpi=300)

    plt.show()

    return results_df
    

def train_and_test(X_train, y_train, X_test, y_test, results_df: pd.DataFrame):
    # Выбор лучшей модели
    best_model_name = results_df.loc[results_df['mean_test_f1'].idxmax(), 'model']
    best_model = models[best_model_name]
    print('Best model name: ', best_model_name)

    # Обучение лучшей модели
    best_model.fit(X_train, y_train)

    # Оценка на тестовых данных
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))

    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))

    report = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(tabulate(pd.DataFrame(report).transpose(), headers="keys", tablefmt="grid"))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(tabulate(cm , tablefmt="grid"))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True,
                annot_kws={"size": 18}, fmt='d', cmap='PiYG', 
                xticklabels=['Ab', 'Wt'], 
                yticklabels=['Ab', 'Wt'])
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

    if y_proba is not None:
        print("\nROC-AUC score:", roc_auc_score(y_test, y_proba))

    def get_feature_importances(model):
        if isinstance(model, Pipeline):
            estimator = model.named_steps['clf']
        else:
            estimator = model
        
        if hasattr(estimator, 'feature_importances_'):
            return estimator.feature_importances_
        
        elif hasattr(estimator, 'coef_'):
            return np.abs(estimator.coef_[0])
        
        else:
            result = permutation_importance(
                model, X_test, y_test,
                n_repeats=10,
                random_state=42
            )
            return result.importances_mean

    importances = get_feature_importances(best_model)

    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15], color = '#fd89c3')
    plt.title('Важность признаков', fontsize=18)
    plt.xlabel('Важность', fontsize=18)
    plt.ylabel('Признаки', fontsize=18)
    plt.gca().invert_yaxis()   

    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14) 

    plt.savefig('feature_importance.png', dpi=300)

    plt.show()

def get_data(path1, path2 = None):
    if path2 is not None:
        df = pd.read_csv(path1) 
        X_train = df.drop(['Name','Type',
                        'DBSCAN_class_11','DBSCAN_class_12','DBSCAN_class_13','DBSCAN_class_14',
                        'DBSCAN_class_21','DBSCAN_class_22','DBSCAN_class_23','DBSCAN_class_24',
                        'DBSCAN_class_31','DBSCAN_class_32','DBSCAN_class_33','DBSCAN_class_34',
                        'DBSCAN_class_41','DBSCAN_class_42','DBSCAN_class_43','DBSCAN_class_44',
                        'DBSCAN_cluster_11','DBSCAN_cluster_12','DBSCAN_cluster_13','DBSCAN_cluster_14','DBSCAN_cluster_15','DBSCAN_cluster_16',
                        'DBSCAN_cluster_21','DBSCAN_cluster_22','DBSCAN_cluster_23','DBSCAN_cluster_24','DBSCAN_cluster_25','DBSCAN_cluster_26',
                        'DBSCAN_cluster_31','DBSCAN_cluster_32','DBSCAN_cluster_33','DBSCAN_cluster_34','DBSCAN_cluster_35','DBSCAN_cluster_36',
                        'DBSCAN_cluster_41','DBSCAN_cluster_42','DBSCAN_cluster_43','DBSCAN_cluster_44','DBSCAN_cluster_45','DBSCAN_cluster_46',
                        'DBSCAN_cluster_51','DBSCAN_cluster_52','DBSCAN_cluster_53','DBSCAN_cluster_54','DBSCAN_cluster_55','DBSCAN_cluster_56',
                        'DBSCAN_cluster_61','DBSCAN_cluster_62','DBSCAN_cluster_63','DBSCAN_cluster_64','DBSCAN_cluster_65','DBSCAN_cluster_66',
                        'N_class_11','N_class_12','N_class_13','N_class_14',
                        'N_class_21','N_class_22','N_class_23','N_class_24',
                        'N_class_31','N_class_32','N_class_33','N_class_34',
                        'N_class_41','N_class_42','N_class_43','N_class_44',
                        'N_cluster_11','N_cluster_12','N_cluster_13','N_cluster_14','N_cluster_15','N_cluster_16',
                        'N_cluster_21','N_cluster_22','N_cluster_23','N_cluster_24','N_cluster_25','N_cluster_26',
                        'N_cluster_31','N_cluster_32','N_cluster_33','N_cluster_34','N_cluster_35','N_cluster_36',
                        'N_cluster_41','N_cluster_42','N_cluster_43','N_cluster_44','N_cluster_45','N_cluster_46',
                        'N_cluster_51','N_cluster_52','N_cluster_53','N_cluster_54','N_cluster_55','N_cluster_56',
                        'N_cluster_61','N_cluster_62','N_cluster_63','N_cluster_64','N_cluster_65','N_cluster_66'], axis=1)  
        y_train = df['Type'] 

        df2 = pd.read_csv(path2) 
        # df2 = df2[df2['Type'] != 'Ab'] # для второго набора диких мышей
        # df2['Type'] = df2['Type'].replace('Ab', 'Wt') # для 9009
        X_test = df2.drop(['Name','Type'], axis=1)  
        y_test = df2['Type'] 

        # label_encoder = LabelEncoder()
        # y_train = label_encoder.fit_transform(y_train)
        # y_test = label_encoder.transform(y_test)
    
    else:
        df = pd.read_csv('dendr_class/all_dendr_metrics.csv') 
        # X = df.drop(['Name','Type'], axis=1)  
        X = df.drop(['Name','Type',
                        'DBSCAN_class_11','DBSCAN_class_12','DBSCAN_class_13','DBSCAN_class_14',
                        'DBSCAN_class_21','DBSCAN_class_22','DBSCAN_class_23','DBSCAN_class_24',
                        'DBSCAN_class_31','DBSCAN_class_32','DBSCAN_class_33','DBSCAN_class_34',
                        'DBSCAN_class_41','DBSCAN_class_42','DBSCAN_class_43','DBSCAN_class_44',
                        'DBSCAN_cluster_11','DBSCAN_cluster_12','DBSCAN_cluster_13','DBSCAN_cluster_14','DBSCAN_cluster_15','DBSCAN_cluster_16',
                        'DBSCAN_cluster_21','DBSCAN_cluster_22','DBSCAN_cluster_23','DBSCAN_cluster_24','DBSCAN_cluster_25','DBSCAN_cluster_26',
                        'DBSCAN_cluster_31','DBSCAN_cluster_32','DBSCAN_cluster_33','DBSCAN_cluster_34','DBSCAN_cluster_35','DBSCAN_cluster_36',
                        'DBSCAN_cluster_41','DBSCAN_cluster_42','DBSCAN_cluster_43','DBSCAN_cluster_44','DBSCAN_cluster_45','DBSCAN_cluster_46',
                        'DBSCAN_cluster_51','DBSCAN_cluster_52','DBSCAN_cluster_53','DBSCAN_cluster_54','DBSCAN_cluster_55','DBSCAN_cluster_56',
                        'DBSCAN_cluster_61','DBSCAN_cluster_62','DBSCAN_cluster_63','DBSCAN_cluster_64','DBSCAN_cluster_65','DBSCAN_cluster_66',
                        'N_class_11','N_class_12','N_class_13','N_class_14',
                        'N_class_21','N_class_22','N_class_23','N_class_24',
                        'N_class_31','N_class_32','N_class_33','N_class_34',
                        'N_class_41','N_class_42','N_class_43','N_class_44',
                        'N_cluster_11','N_cluster_12','N_cluster_13','N_cluster_14','N_cluster_15','N_cluster_16',
                        'N_cluster_21','N_cluster_22','N_cluster_23','N_cluster_24','N_cluster_25','N_cluster_26',
                        'N_cluster_31','N_cluster_32','N_cluster_33','N_cluster_34','N_cluster_35','N_cluster_36',
                        'N_cluster_41','N_cluster_42','N_cluster_43','N_cluster_44','N_cluster_45','N_cluster_46',
                        'N_cluster_51','N_cluster_52','N_cluster_53','N_cluster_54','N_cluster_55','N_cluster_56',
                        'N_cluster_61','N_cluster_62','N_cluster_63','N_cluster_64','N_cluster_65','N_cluster_66'], axis=1)  
        y = df['Type'] 

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.25, 
            stratify=y,
            random_state=42
        )

        print("Количество в y_train:")
        print(y_train.value_counts())

        print("\nКоличество в y_test:")
        print(y_test.value_counts())

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test
