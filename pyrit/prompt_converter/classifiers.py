from typing import Optional, Union, Dict

import torch
import numpy as np
from torch import nn
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder, losses
from datasets import Dataset
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from setfit import SetFitModel, SetFitHead, SetFitTrainer

from pyrit.prompt_converter.utils import st_cache_data

class RidgeClassifierCVProb(RidgeClassifierCV):
    def predict_proba(self, X):
        if len(self.classes_) == 1:
            return np.ones((len(X), 1))
        d = self.decision_function(X)
        if len(self.classes_) == 2:
            probs = np.exp(d) / (np.exp(d) + np.exp(-d))
            return np.array([1 - probs, probs]).T
        probs = np.exp(d).T / np.sum(np.exp(d), axis=1)
        return probs.T


class ClaimClassifierBase:
    """
    Base class with public methods
    
    Child classes should implement private memoized _prep_data, _fit, and _predict methods,
    while the streamlit app calls these public methods
    """
    def __init__(self, predict_without_fit=False):
        self.predict_without_fit = predict_without_fit
        self._is_fitted = False

        # The counter keeps track of the number of times a model has been fit.
        # Child methods are wrapped with `st.cache_data`, and we only
        # want them to run if the passed data or model parameters have changed
        self._fit_counter = 0

    def prep_data(self, gen_df, remove_claims_with_homogenous_label=True, rebalance=True):
        gen_df.sort_index(inplace=True)
        return self._prep_data(gen_df, remove_claims_with_homogenous_label, rebalance)

    def fit(self, train_df):
        """Up to child method to update `_is_fitted` and `_fit_counter`"""
        train_df.sort_index(inplace=True)
        self._fit(train_df) # data+counter triggers memoization

    def predict(self, test_df):
        test_df.sort_index(inplace=True)
        return self._predict(test_df, self._fit_counter) # data+counter triggers memoization

    @staticmethod
    def _split_data(gen_df, remove_claims_with_homogenous_label=True, rebalance=True):
        """
        Split train/test data, optionally removing claims with only one label and/or rebalancing
        """
        labeled = gen_df["label"].notna()
        train_df, test_df = gen_df.loc[labeled], gen_df.loc[~labeled]
        if remove_claims_with_homogenous_label:
            train_df = (
                train_df.groupby("claim", as_index=False, sort=False)
                        .filter(lambda x: x["label"].nunique()>1)
            )
        if rebalance:
            n_pos = train_df.label.sum()
            n_neg = len(train_df) - n_pos
            train_df = (
                train_df.groupby("label", as_index=False, sort=False)
                        .head(min(n_pos, n_neg))
            )
        return train_df, test_df


class ClaimClassifierCE(ClaimClassifierBase):
    def __init__(
        self,
        model_type="cross-encoder/nli-deberta-v3-base",
        predict_without_fit=False,
        classifier_class=RidgeClassifierCVProb,
        classifier_kwargs={},
        cache_dir=None,
    ):
        self.encoder = CrossEncoder(model_type, automodel_args={"cache_dir": cache_dir})
        self.classifier_class = classifier_class
        self.classifier_kwargs = classifier_kwargs

        self.label2id = self.encoder.config.label2id
        self.labels = list(self.label2id.keys())
        self.ids = list(self.label2id.values())
        super().__init__(predict_without_fit=predict_without_fit)

    @st_cache_data(show_spinner="Preparing data for classification...")
    def _prep_data(_self, gen_df, remove_claims_with_homogenous_label=True, rebalance=True):
        """
        Process the annotation dataframe
        Returns: training dataframe of encoder outputs, training labels, and
        test encoder outputs
        """
        self = _self
        gen_df = gen_df.copy()
        assert {"claim", "inst", "label"}.issubset(gen_df.columns)
        ce_probs = self.encoder.predict(gen_df[["claim", "inst"]].values.tolist())
        gen_df[self.labels] = ce_probs[:, self.ids]
        train_df, test_df = self._split_data(
            gen_df, remove_claims_with_homogenous_label, rebalance=rebalance
        )
        return train_df, test_df

    @st_cache_data(show_spinner="Fitting failure classifier...")
    def _fit(_self, train_df):
        self = _self
        if len(train_df) > 0: # do not fit if no labels
            assert {"label", *self.labels}.issubset(train_df.columns)
            self.classifier = self.classifier_class(**self.classifier_kwargs)
            train_ce_p = train_df[self.labels].values
            train_y = train_df['label'].values
            self.classifier.fit(train_ce_p, train_y)
            self._is_fitted = True
            self._fit_counter += 1

    @st_cache_data(show_spinner="Predicting likely failures...")
    def _predict(_self, test_df, fit_counter=None):
        self = _self
        test_ce_p = test_df[self.labels].values
        if self._is_fitted:
            preds = self.classifier.predict(test_ce_p)
            probs = self.classifier.predict_proba(test_ce_p)
            pos_probs = probs[:, self.classifier.classes_.tolist().index(1)]
        elif self.predict_without_fit: # test_x is predicted probabilities from crossencoder
            preds = test_df["entailment"] > test_df["contradiction"]
            pos_probs = softmax(
                test_df[["entailment", "contradiction"]].values, axis=1
            )[:, 0]
        else:
            raise ValueError("`self._fit()` has not been called")
        return pos_probs, preds


class ClaimClassifierSF(ClaimClassifierBase):
    """
    Claim Classifier initialized from a `transformers.AutoModelForSequenceClassification`
    pretrained on NLI data. Model is fine-tuned using few-shot SetFit method.
    """
    def __init__(
        self,
        model_type="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        use_differentiable_head=False,
        loss_class=losses.CosineSimilarityLoss,
        num_iterations=20,
        body_num_epochs=1,
        body_learning_rate=0.00001,
        head_num_epochs=10,
        head_learning_rate=0.01,
        batch_size=20,
        refit_on_update=True,
        predict_without_fit=False,
        cache_dir=None,
    ):
        super().__init__(predict_without_fit)
        self.model = SetFitModelFromClassifier.from_pretrained(
            model_type,
            use_differentiable_head=use_differentiable_head,
            sklearn_classifier=RidgeClassifierCVProb, # only used if not differentiable head
            cache_dir=cache_dir,
        )
        self.loss_class = loss_class
        self.num_iterations = num_iterations
        self.body_num_epochs = body_num_epochs
        self.body_learning_rate = body_learning_rate
        self.head_num_epochs = head_num_epochs
        self.head_learning_rate = head_learning_rate
        self.batch_size = batch_size
        self.refit_on_update = refit_on_update

        self.use_differentiable_head = isinstance(self.model.model_head, nn.Module)
        body_config = self.model.model_body[0].auto_model.config
        if self.use_differentiable_head:
            self.label2id = body_config.label2id
        else:
            self.label2id = {"entailment": 1, "contradiction": 0}
        self.annotation2id = { # covering bases
            1: self.label2id["entailment"],
            0: self.label2id["contradiction"],
        }
        if body_config.model_type in ["deberta-v2", "albert"]:
            self.sent_sep = "[SEP]"
        elif body_config.model_type == "roberta":
            self.sent_sep = "</s></s>"
        else:
            self.sent_sep = "[SEP]"

    @st_cache_data(show_spinner=False)
    def _prep_data(_self, gen_df, remove_claims_with_homogenous_label=True, rebalance=True):
        """
        Process the annotation dataframe,
        returns training datasets.Dataset, None (for API reasons), and list of test texts
        """
        self = _self
        gen_df = gen_df.copy()
        assert {"claim", "inst", "label"}.issubset(gen_df.columns)
        gen_df["text"] = gen_df["claim"] + self.sent_sep + gen_df["inst"]
        train_df, test_df = self._split_data(
            gen_df, remove_claims_with_homogenous_label, rebalance
        )
        train_df = train_df.assign(
            label=lambda x: x['label'].astype(int).replace(self.annotation2id)
        )
        return train_df, test_df

    @st_cache_data(show_spinner="Fitting failure classifier...")
    def _fit(_self, train_df):
        """`train_df` is a `pd.DataFrame`. `fit_counter` used to trigger memoization"""
        self = _self
        assert {"text", "label"}.issubset(train_df.columns)
        if len(train_df) == 0:
            return
        if self.refit_on_update and self._is_fitted:
            self.model._reset_parameters()
            self._is_fitted = False
        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=Dataset.from_pandas(train_df[["text", "label"]]),
            batch_size=min(len(train_df), self.batch_size),
            loss_class=self.loss_class,
            num_iterations=self.num_iterations,
            num_epochs=self.body_num_epochs,
            learning_rate=self.body_learning_rate,
        )
        if self.use_differentiable_head:
            # must unfreeze/freeze to account for additional training w/ same model
            trainer.unfreeze() # unfreezes head, body
            trainer.freeze() # freezes head, not body
            trainer.train() # trains body
            trainer.unfreeze(keep_body_frozen=True) # freezes body
            trainer.train(
                num_epochs=self.head_num_epochs,
                learning_rate=self.head_learning_rate,
                l2_weight=0.0,
            )
        else:
            trainer.train()
        self._is_fitted = True
        self._fit_counter += 1

    @st_cache_data(show_spinner="Predicting likely failures...")
    def _predict(_self, test_df, fit_counter=None):
        """`test_df` is a `pd.DataFrame`. `fit_counter` used to trigger memoization"""
        self = _self
        assert "text" in test_df.columns
        test_x = test_df["text"].values.tolist()
        if self._is_fitted:
            # both pulled more-or-less from SetFitModel.predict,
            with torch.no_grad(): 
                probs = self.model.predict_proba(test_x)
                if isinstance(probs, torch.Tensor):
                    probs = probs.cpu().numpy()
        elif self.predict_without_fit:
            # the following is a bit inelegant and slow (due to device copying),
            # but it shouldn't occur often since usually _fit() has been called
            with torch.no_grad():
                embeddings = self.model.model_body.encode(test_x, convert_to_tensor=True)
                w = self.model.model_body._classifier_weight.to(embeddings.device)
                b = self.model.model_body._classifier_bias.to(embeddings.device)
                probs = torch.softmax(nn.functional.linear(embeddings, w, b), axis=1)
                probs = probs.cpu().numpy()
        else:
            raise ValueError("`self._fit()` has not been called")

        if len(probs.shape) == 1:
            preds = probs > 0.5
            pos_probs = probs
        else:
            pos_idx, neg_idx = self.label2id["entailment"], self.label2id["contradiction"]
            preds = probs[:, pos_idx] > probs[:, neg_idx]
            pos_probs = (
                probs[:, pos_idx] / probs[:, [pos_idx, neg_idx]].sum(1)
            )
        return pos_probs, preds



class TransformerForSequenceClassification(Transformer):
    def _load_model(self, model_name_or_path, config, cache_dir, **model_args):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )
        if model.config.model_type in ["deberta-v2", "albert"]:
            self._classifier_weight = model.classifier.weight.detach()
            self._classifier_bias = model.classifier.bias.detach()
            model.classifier = nn.Identity() # overwrite
        elif model.config.model_type == "roberta":
            self._classifier_weight = model.classifier.out_proj.weight.detach()
            self._classifier_bias = model.classifier.out_proj.bias.detach()
            model.classifier.out_proj = nn.Identity() # overwrite
        else:
            raise NotImplementedError(f"only deberta-v2 and roberta supported, you used {model.config.model_type}")
        self.auto_model = model
    
    def forward(self, features):
        features = super().forward(features)
        embeds = features["token_embeddings"] # already pooled
        features["cls_token_embeddings"] = embeds
        features["token_embeddings"] = embeds.unsqueeze(1)
        return features


class SentenceTransformerFromClassifier(SentenceTransformer):
    def _load_auto_model(self, model_name_or_path):
        transformer_model = TransformerForSequenceClassification(model_name_or_path)#TODO: , tokenizer_args={"use_fast": False})
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        self._classifier_weight = transformer_model._classifier_weight
        self._classifier_bias = transformer_model._classifier_bias
        return [transformer_model, pooling_model]


class SetFitModelFromClassifier(SetFitModel):
    """
    Initialize a SetFitModel from an existing pretrained AutoModelForSequenceClassification
    """
    @classmethod
    def _from_pretrained(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: Optional[bool] = None,
        proxies: Optional[Dict] = None,
        resume_download: Optional[bool] = None,
        local_files_only: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        multi_target_strategy: Optional[str] = None,
        sklearn_classifier = None,
        use_differentiable_head: bool = True,
        **model_kwargs,
    ):
        model_body = SentenceTransformerFromClassifier(model_id, cache_folder=cache_dir)
        if use_differentiable_head:
            weight = model_body._classifier_weight
            bias = model_body._classifier_bias
            device = model_body._target_device
            model_head = SetFitHead(
                in_features=weight.shape[1],
                out_features=weight.shape[0], 
                device=device,
            )
            model_head.linear.weight.data.copy_(weight)
            model_head.linear.bias.data.copy_(bias)
            if hasattr(model_head, "predict_prob"): #bugfix before new SetFit release
                model_head.predict_proba = model_head.predict_prob
        else:
            if sklearn_classifier is None:
                sklearn_classifier = LogisticRegression
            if "head_params" in model_kwargs.keys():
                clf = sklearn_classifier(**model_kwargs["head_params"])
            else:
                clf = sklearn_classifier()
            if multi_target_strategy is not None:
                if multi_target_strategy == "one-vs-rest":
                    multilabel_classifier = OneVsRestClassifier(clf)
                elif multi_target_strategy == "multi-output":
                    multilabel_classifier = MultiOutputClassifier(clf)
                elif multi_target_strategy == "classifier-chain":
                    multilabel_classifier = ClassifierChain(clf)
                else:
                    raise ValueError(f"multi_target_strategy {multi_target_strategy} is not supported.")

                model_head = multilabel_classifier
            else:
                model_head = clf

        return SetFitModelFromClassifier(
            model_body=model_body,
            model_head=model_head,
            multi_target_strategy=multi_target_strategy,
        )

    def predict_proba(self, x_test: torch.Tensor) -> torch.Tensor:
        to_tensor = False
        if isinstance(self.model_head, nn.Module):
            to_tensor = True
        embeddings = self.model_body.encode(x_test, convert_to_tensor=to_tensor)
        return self.model_head.predict_proba(embeddings)

    def _reset_parameters(self):
        # self.model_body.load_state_dict(self.model_original_state)
        if isinstance(self.model_head, nn.Module):
            weight = self.model_body._classifier_weight
            bias = self.model_body._classifier_bias
            self.model_head.linear.weight.data.copy_(weight)
            self.model_head.linear.bias.data.copy_(bias)


def fit_and_predict(claim_classifier, gen_df, do_fit=True):
    """
    Fit the claim classifier and predict unseen labels
    """
    assert {"claim", "inst", "label"}.issubset(gen_df.columns)
    # all the below methods are memoized and will only run if either the passed data
    # have changed or the `claim_classifier` has been refit

    # only provide necessary columns to avoid cache trigger
    # Exclude labels w/ 2 (created by `build_annotator_radio`, if used) that
    # indicate an unrelated failure. Can incorporate this a later date
    train_df, test_df = claim_classifier.prep_data(
        gen_df[["claim", "inst", "label"]].loc[gen_df.label != 2],
        remove_claims_with_homogenous_label=False,
        rebalance=True, # rebalancing may not be necessary with setfit
    )

    if do_fit and len(test_df) > 0: # don't re-fit if there is nothing to predict on
        claim_classifier.fit(train_df)
    if len(test_df) > 0:
        probs, preds = claim_classifier.predict(test_df)

    # add predictions to the data; some train_df could have been dropped 
    # because of `remove_claims_with_homogenous_labels``, so we do not concat `train_df`
    gen_df["prob"] = gen_df["label"].astype(float)
    gen_df["pred"] = gen_df["label"]
    if len(test_df) > 0:
        gen_df.loc[test_df.index, "prob"] = probs
        gen_df.loc[test_df.index, "pred"] = preds
    return gen_df