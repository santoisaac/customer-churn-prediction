{
  "metadata": {
    "kernelspec": {
      "language": "python",
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.11.11",
      "mimetype": "text/x-python",
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "pygments_lexer": "ipython3",
      "nbconvert_exporter": "python",
      "file_extension": ".py"
    },
    "kaggle": {
      "accelerator": "none",
      "dataSources": [
        {
          "sourceId": 3322096,
          "sourceType": "datasetVersion",
          "datasetId": 2008274
        }
      ],
      "dockerImageVersionId": 31040,
      "isInternetEnabled": true,
      "language": "python",
      "sourceType": "notebook",
      "isGpuEnabled": false
    },
    "colab": {
      "provenance": [],
      "include_colab_link": true
    }
  },
  "nbformat_minor": 0,
  "nbformat": 4,
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/maanjadhav/Customer-Churn-Prediction/blob/main/Copy_of_Customer_Churn_Prediction.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "source": [
        "# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,\n",
        "# THEN FEEL FREE TO DELETE THIS CELL.\n",
        "# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON\n",
        "# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR\n",
        "# NOTEBOOK.\n",
        "import kagglehub\n",
        "shantanudhakadd_bank_customer_churn_prediction_path = kagglehub.dataset_download('shantanudhakadd/bank-customer-churn-prediction')\n",
        "\n",
        "print('Data source import complete.')\n"
      ],
      "metadata": {
        "id": "XdJZpWrv0zeF"
      },
      "cell_type": "code",
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "# This Python 3 environment comes with many helpful analytics libraries installed\n",
        "# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n",
        "# For example, here's several helpful packages to load\n",
        "\n",
        "import numpy as np # linear algebra\n",
        "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
        "\n",
        "# Input data files are available in the read-only \"../input/\" directory\n",
        "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
        "\n",
        "import os\n",
        "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
        "    for filename in filenames:\n",
        "        print(os.path.join(dirname, filename))\n",
        "\n",
        "# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\"\n",
        "# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session"
      ],
      "metadata": {
        "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
        "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
        "trusted": true,
        "id": "_5sSsFfW0zeH"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T16:59:56.280518Z",
          "iopub.execute_input": "2025-06-24T16:59:56.28083Z",
          "iopub.status.idle": "2025-06-24T16:59:58.374738Z",
          "shell.execute_reply.started": "2025-06-24T16:59:56.280804Z",
          "shell.execute_reply": "2025-06-24T16:59:58.373767Z"
        },
        "id": "SxmzmTHg0zeI"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "np.random.seed(42)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:00:40.980252Z",
          "iopub.execute_input": "2025-06-24T17:00:40.98056Z",
          "iopub.status.idle": "2025-06-24T17:00:40.985155Z",
          "shell.execute_reply.started": "2025-06-24T17:00:40.980538Z",
          "shell.execute_reply": "2025-06-24T17:00:40.984194Z"
        },
        "id": "6KB9ScSb0zeJ"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv('/kaggle/input/bank-customer-churn-prediction/Churn_Modelling.csv')"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:02:04.854411Z",
          "iopub.execute_input": "2025-06-24T17:02:04.854747Z",
          "iopub.status.idle": "2025-06-24T17:02:04.884236Z",
          "shell.execute_reply.started": "2025-06-24T17:02:04.85472Z",
          "shell.execute_reply": "2025-06-24T17:02:04.883344Z"
        },
        "id": "vnwavySt0zeK"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "df.info()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:02:15.12359Z",
          "iopub.execute_input": "2025-06-24T17:02:15.123904Z",
          "iopub.status.idle": "2025-06-24T17:02:15.137781Z",
          "shell.execute_reply.started": "2025-06-24T17:02:15.123879Z",
          "shell.execute_reply": "2025-06-24T17:02:15.136349Z"
        },
        "id": "pmQuEc1g0zeL"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull().sum()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:02:26.084557Z",
          "iopub.execute_input": "2025-06-24T17:02:26.084911Z",
          "iopub.status.idle": "2025-06-24T17:02:26.0967Z",
          "shell.execute_reply.started": "2025-06-24T17:02:26.084871Z",
          "shell.execute_reply": "2025-06-24T17:02:26.095379Z"
        },
        "id": "dd0Be1BO0zeL"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:02:33.421482Z",
          "iopub.execute_input": "2025-06-24T17:02:33.421765Z",
          "iopub.status.idle": "2025-06-24T17:02:33.447079Z",
          "shell.execute_reply.started": "2025-06-24T17:02:33.421748Z",
          "shell.execute_reply": "2025-06-24T17:02:33.446265Z"
        },
        "id": "uK_rXPoa0zeM"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "def create_features(df):\n",
        "    df['Balance_to_Salary_Ratio'] = df['Balance'] / df['EstimatedSalary'].clip(lower=1)\n",
        "    df['Tenure_to_Age_Ratio'] = df['Tenure'] / df['Age'].clip(lower=1)\n",
        "    df['CreditScore_Age'] = df['CreditScore'] / df['Age'].clip(lower=1)\n",
        "    return df"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:03:11.212334Z",
          "iopub.execute_input": "2025-06-24T17:03:11.212676Z",
          "iopub.status.idle": "2025-06-24T17:03:11.218663Z",
          "shell.execute_reply.started": "2025-06-24T17:03:11.212653Z",
          "shell.execute_reply": "2025-06-24T17:03:11.217644Z"
        },
        "id": "NbQw04Kl0zeN"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "df = create_features(df)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:03:28.539588Z",
          "iopub.execute_input": "2025-06-24T17:03:28.539883Z",
          "iopub.status.idle": "2025-06-24T17:03:28.560176Z",
          "shell.execute_reply.started": "2025-06-24T17:03:28.539852Z",
          "shell.execute_reply": "2025-06-24T17:03:28.558718Z"
        },
        "id": "fyVyweQW0zeP"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:03:54.372216Z",
          "iopub.execute_input": "2025-06-24T17:03:54.372493Z",
          "iopub.status.idle": "2025-06-24T17:03:54.390328Z",
          "shell.execute_reply.started": "2025-06-24T17:03:54.372475Z",
          "shell.execute_reply": "2025-06-24T17:03:54.389519Z"
        },
        "id": "2Qh8vkud0zeQ"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',\n",
        "                    'EstimatedSalary', 'Balance_to_Salary_Ratio', 'Tenure_to_Age_Ratio',\n",
        "                    'CreditScore_Age']\n",
        "categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']\n",
        "X = df[numeric_features + categorical_features]\n",
        "y = df['Exited']"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:04:31.432514Z",
          "iopub.execute_input": "2025-06-24T17:04:31.4329Z",
          "iopub.status.idle": "2025-06-24T17:04:31.444028Z",
          "shell.execute_reply.started": "2025-06-24T17:04:31.432871Z",
          "shell.execute_reply": "2025-06-24T17:04:31.443077Z"
        },
        "id": "X0jlNj6Z0zeR"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "numeric_transformer = StandardScaler()\n",
        "categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:04:51.327625Z",
          "iopub.execute_input": "2025-06-24T17:04:51.327993Z",
          "iopub.status.idle": "2025-06-24T17:04:51.333739Z",
          "shell.execute_reply.started": "2025-06-24T17:04:51.327969Z",
          "shell.execute_reply": "2025-06-24T17:04:51.332597Z"
        },
        "id": "6phR6oEO0zeR"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "preprocessor = ColumnTransformer(\n",
        "    transformers=[\n",
        "        ('num', numeric_transformer, numeric_features),\n",
        "        ('cat', categorical_transformer, categorical_features)\n",
        "    ])"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:05:08.148585Z",
          "iopub.execute_input": "2025-06-24T17:05:08.148893Z",
          "iopub.status.idle": "2025-06-24T17:05:08.15535Z",
          "shell.execute_reply.started": "2025-06-24T17:05:08.148873Z",
          "shell.execute_reply": "2025-06-24T17:05:08.153992Z"
        },
        "id": "nEZ_1LHJ0zeR"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:05:31.16989Z",
          "iopub.execute_input": "2025-06-24T17:05:31.170703Z",
          "iopub.status.idle": "2025-06-24T17:05:31.185728Z",
          "shell.execute_reply.started": "2025-06-24T17:05:31.170674Z",
          "shell.execute_reply": "2025-06-24T17:05:31.184708Z"
        },
        "id": "zP_m-wuA0zeS"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "X_train.shape"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:05:53.019714Z",
          "iopub.execute_input": "2025-06-24T17:05:53.020363Z",
          "iopub.status.idle": "2025-06-24T17:05:53.026523Z",
          "shell.execute_reply.started": "2025-06-24T17:05:53.020332Z",
          "shell.execute_reply": "2025-06-24T17:05:53.025524Z"
        },
        "id": "aZIiJdvs0zeS"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "X_test.shape"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:06:11.600504Z",
          "iopub.execute_input": "2025-06-24T17:06:11.600818Z",
          "iopub.status.idle": "2025-06-24T17:06:11.606819Z",
          "shell.execute_reply.started": "2025-06-24T17:06:11.600794Z",
          "shell.execute_reply": "2025-06-24T17:06:11.60575Z"
        },
        "id": "ssvedWoo0zeS"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "models = {\n",
        "    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),\n",
        "    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),\n",
        "    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)\n",
        "}"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:06:37.18853Z",
          "iopub.execute_input": "2025-06-24T17:06:37.188945Z",
          "iopub.status.idle": "2025-06-24T17:06:37.195512Z",
          "shell.execute_reply.started": "2025-06-24T17:06:37.188906Z",
          "shell.execute_reply": "2025-06-24T17:06:37.194535Z"
        },
        "id": "zHJ_WXoK0zeT"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "results = []\n",
        "feature_importance = {}"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:09:05.527639Z",
          "iopub.execute_input": "2025-06-24T17:09:05.528286Z",
          "iopub.status.idle": "2025-06-24T17:09:05.532355Z",
          "shell.execute_reply.started": "2025-06-24T17:09:05.528259Z",
          "shell.execute_reply": "2025-06-24T17:09:05.531187Z"
        },
        "id": "z1AD4ol80zeT"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "for name, model in models.items():\n",
        "    pipeline = Pipeline([\n",
        "        ('preprocessor', preprocessor),\n",
        "        ('classifier', model)\n",
        "    ])\n",
        "    pipeline.fit(X_train, y_train)\n",
        "    y_pred = pipeline.predict(X_test)\n",
        "    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]\n",
        "    metrics = {\n",
        "        'Model': name,\n",
        "        'Accuracy': accuracy_score(y_test, y_pred),\n",
        "        'Precision': precision_score(y_test, y_pred),\n",
        "        'Recall': recall_score(y_test, y_pred),\n",
        "        'F1 Score': f1_score(y_test, y_pred),\n",
        "        'ROC AUC': roc_auc_score(y_test, y_pred_proba)\n",
        "    }\n",
        "    results.append(metrics)\n",
        "    if name in ['Random Forest', 'Gradient Boosting']:\n",
        "        feature_names = (numeric_features +\n",
        "                         list(pipeline.named_steps['preprocessor']\n",
        "                             .named_transformers_['cat']\n",
        "                             .get_feature_names_out(categorical_features)))\n",
        "        feature_importance[name] = pd.DataFrame({\n",
        "            'Feature': feature_names,\n",
        "            'Importance': model.feature_importances_\n",
        "        }).sort_values('Importance', ascending=False)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:09:44.957574Z",
          "iopub.execute_input": "2025-06-24T17:09:44.957899Z",
          "iopub.status.idle": "2025-06-24T17:09:48.61265Z",
          "shell.execute_reply.started": "2025-06-24T17:09:44.957876Z",
          "shell.execute_reply": "2025-06-24T17:09:48.611749Z"
        },
        "id": "ax_H90Ov0zeT"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "results_df = pd.DataFrame(results)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:10:31.052318Z",
          "iopub.execute_input": "2025-06-24T17:10:31.052641Z",
          "iopub.status.idle": "2025-06-24T17:10:31.058444Z",
          "shell.execute_reply.started": "2025-06-24T17:10:31.052619Z",
          "shell.execute_reply": "2025-06-24T17:10:31.057339Z"
        },
        "id": "8ybBZxZp0zeT"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "results_df.round(4)"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:10:37.973916Z",
          "iopub.execute_input": "2025-06-24T17:10:37.974251Z",
          "iopub.status.idle": "2025-06-24T17:10:37.985347Z",
          "shell.execute_reply.started": "2025-06-24T17:10:37.974228Z",
          "shell.execute_reply": "2025-06-24T17:10:37.984411Z"
        },
        "id": "O9IjD_jM0zeU"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(10, 6))\n",
        "sns.barplot(data=feature_importance['Random Forest'].head(10),\n",
        "            x='Importance', y='Feature')\n",
        "plt.title('Top 10 Feature Importance (Random Forest)')\n",
        "plt.tight_layout()\n",
        "plt.savefig('bank_feature_importance.png')\n",
        "plt.show()"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:11:07.110872Z",
          "iopub.execute_input": "2025-06-24T17:11:07.111288Z",
          "iopub.status.idle": "2025-06-24T17:11:07.663305Z",
          "shell.execute_reply.started": "2025-06-24T17:11:07.111264Z",
          "shell.execute_reply": "2025-06-24T17:11:07.662296Z"
        },
        "id": "oINDlh_U0zeU"
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "results_df.to_csv('bank_model_performance.csv', index=False)\n",
        "print(\"\\nResults saved to 'bank_model_performance.csv'\")\n",
        "print(\"Feature importance plot saved to 'bank_feature_importance.png'\")"
      ],
      "metadata": {
        "trusted": true,
        "execution": {
          "iopub.status.busy": "2025-06-24T17:11:39.82148Z",
          "iopub.execute_input": "2025-06-24T17:11:39.821778Z",
          "iopub.status.idle": "2025-06-24T17:11:39.834655Z",
          "shell.execute_reply.started": "2025-06-24T17:11:39.821759Z",
          "shell.execute_reply": "2025-06-24T17:11:39.833254Z"
        },
        "id": "UmEbF2ym0zeU"
      },
      "outputs": [],
      "execution_count": null
    }
  ]
}
