import os
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import time
import great_expectations as gx


class DataLoader:
    """データロードを行うクラス"""

    @staticmethod
    def load_titanic_data(path=None):
        """Titanicデータセットを読み込む"""
        if path and os.path.exists(path):
            return pd.read_csv(path)
        else:
            # スクリプトのあるディレクトリを基準に相対パスでファイルを探す
            base_dir = os.path.dirname(__file__)  # この.pyファイルのディレクトリ
            local_path = os.path.join(base_dir, "data", "Titanic.csv")
            if os.path.exists(local_path):
                return pd.read_csv(local_path)
            else:
                raise FileNotFoundError(f"Titanic.csv not found at {local_path}")

    @staticmethod
    def preprocess_titanic_data(data):
        """Titanicデータを前処理する"""
        # 必要な特徴量を選択
        data = data.copy()

        # 不要な列を削除
        columns_to_drop = []
        for col in ["PassengerId", "Name", "Ticket", "Cabin"]:
            if col in data.columns:
                columns_to_drop.append(col)

        if columns_to_drop:
            data.drop(columns_to_drop, axis=1, inplace=True)

        # 目的変数とその他を分離
        if "Survived" in data.columns:
            y = data["Survived"]
            X = data.drop("Survived", axis=1)
            return X, y
        else:
            return data, None


class DataValidator:
    """データバリデーションを行うクラス"""

    @staticmethod
    def validate_titanic_data(data):
        """Titanicデータセットの検証"""
        import pandas as pd

        # DataFrameに変換
        if not isinstance(data, pd.DataFrame):
            return False, ["データはpd.DataFrameである必要があります"]

        try:
            context = get_context()

            # データソースの設定（明示的に）
            datasource_config = {
                "name": "my_pandas_datasource",
                "class_name": "Datasource",
                "execution_engine": {"class_name": "PandasExecutionEngine"},
                "data_connectors": {
                    "runtime_data_connector": {
                        "class_name": "RuntimeDataConnector",
                        "batch_identifiers": ["default_identifier_name"]
                    }
                }
            }
            context.add_or_update_datasource(**datasource_config)

            # バッチリクエスト作成
            batch_request = BatchRequest(
                datasource_name="my_pandas_datasource",
                data_connector_name="runtime_data_connector",
                data_asset_name="my_asset",
                runtime_parameters={"batch_data": data},
                batch_identifiers={"default_identifier_name": "default_id"},
            )

            # Validator取得
            validator = context.get_validator(batch_request=batch_request)

            # 必須カラムチェック
            required_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                print(f"警告: 以下のカラムがありません: {missing_columns}")
                return False, [{"success": False, "missing_columns": missing_columns}]

            # バリデーションルール
            results = []

            results.append(
                validator.expect_column_distinct_values_to_be_in_set(
                    column="Pclass", value_set=[1, 2, 3]
                )
            )
            results.append(
                validator.expect_column_distinct_values_to_be_in_set(
                    column="Sex", value_set=["male", "female"]
                )
            )
            results.append(
                validator.expect_column_values_to_be_between(
                    column="Age", min_value=0, max_value=100
                )
            )
            results.append(
                validator.expect_column_values_to_be_between(
                    column="Fare", min_value=0, max_value=600
                )
            )
            results.append(
                validator.expect_column_distinct_values_to_be_in_set(
                    column="Embarked", value_set=["C", "Q", "S", ""]
                )
            )

            # 出力と成功判定
            for result in results:
                print(f"🧪 {result.expectation_config.expectation_type} → {'✅ OK' if result.success else '❌ NG'}")

            is_successful = all(r.success for r in results)
            return is_successful, results

        except Exception as e:
            print(f"Great Expectations検証エラー: {e}")
            return False, [{"success": False, "error": str(e)}]


class ModelTester:
    """モデルテストを行うクラス"""

    @staticmethod
    def create_preprocessing_pipeline():
        """前処理パイプラインを作成"""
        numeric_features = ["Age", "Fare", "SibSp", "Parch"]
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_features = ["Pclass", "Sex", "Embarked"]
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",  # 指定されていない列は削除
        )
        return preprocessor

    @staticmethod
    def train_model(X_train, y_train, model_params=None):
        """モデルを学習する"""
        if model_params is None:
            model_params = {"n_estimators": 100, "random_state": 42}

        # 前処理パイプラインを作成
        preprocessor = ModelTester.create_preprocessing_pipeline()

        # モデル作成
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(**model_params)),
            ]
        )

        # 学習
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """モデルを評価する"""
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy, "inference_time": inference_time}

    @staticmethod
    def save_model(model, path="models/titanic_model.pkl"):
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"titanic_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        return path

    @staticmethod
    def load_model(path="models/titanic_model.pkl"):
        """モデルを読み込む"""
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def compare_with_baseline(current_metrics, baseline_threshold=0.75):
        """ベースラインと比較する"""
        return current_metrics["accuracy"] >= baseline_threshold


# テスト関数（pytestで実行可能）
def test_data_validation():
    """データバリデーションのテスト"""
    # データロード
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)

    # 正常なデータのチェック
    success, results = DataValidator.validate_titanic_data(X)
    assert success, "データバリデーションに失敗しました"

    # 異常データのチェック
    bad_data = X.copy()
    bad_data.loc[0, "Pclass"] = 5  # 明らかに範囲外の値
    success, results = DataValidator.validate_titanic_data(bad_data)
    assert not success, "異常データをチェックできませんでした"


def test_model_performance():
    """モデル性能のテスト"""
    # データ準備
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデル学習
    model = ModelTester.train_model(X_train, y_train)

    # 評価
    metrics = ModelTester.evaluate_model(model, X_test, y_test)

    # ベースラインとの比較
    assert ModelTester.compare_with_baseline(
        metrics, 0.75
    ), f"モデル性能がベースラインを下回っています: {metrics['accuracy']}"

    # 推論時間の確認
    assert (
        metrics["inference_time"] < 1.0
    ), f"推論時間が長すぎます: {metrics['inference_time']}秒"


if __name__ == "__main__":
    # データロード
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)

    # データバリデーション
    success, results = DataValidator.validate_titanic_data(X)
    print(f"データ検証結果: {'成功' if success else '失敗'}")
    for result in results:
        # "success": falseの場合はエラーメッセージを表示
        if not result["success"]:
            print(f"異常タイプ: {result['expectation_config']['type']}, 結果: {result}")
    if not success:
        print("データ検証に失敗しました。処理を終了します。")
        exit(1)

    # モデルのトレーニングと評価
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # パラメータ設定
    model_params = {"n_estimators": 100, "random_state": 42}

    # モデルトレーニング
    model = ModelTester.train_model(X_train, y_train, model_params)
    metrics = ModelTester.evaluate_model(model, X_test, y_test)

    print(f"精度: {metrics['accuracy']:.4f}")
    print(f"推論時間: {metrics['inference_time']:.4f}秒")

    # モデル保存
    model_path = ModelTester.save_model(model)

    # ベースラインとの比較
    baseline_ok = ModelTester.compare_with_baseline(metrics)
    print(f"ベースライン比較: {'合格' if baseline_ok else '不合格'}")


def test_inference_speed_and_accuracy():
    # データの読み込みと前処理
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)

    # データ検証
    success, _ = DataValidator.validate_titanic_data(X)
    assert success, "データ検証に失敗しました"

    # 訓練/テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルのトレーニング
    model_params = {"n_estimators": 100, "random_state": 42}
    model = ModelTester.train_model(X_train, y_train, model_params)

    # 推論時間と精度の測定
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    inference_time = end_time - start_time

    # 結果の出力（オプション）
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")

    # 検証: 例として精度は0.7以上、推論時間は1秒未満であることを確認
    assert accuracy >= 0.7, f"精度が低すぎます: {accuracy}"
    assert (
        inference_time < 1.0
    ), f"推論に時間がかかりすぎています: {inference_time:.4f}秒"
