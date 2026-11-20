"""
Data preparation for the Parkinson's resting-state EEG project.

Builds the unified dataset (`full_df`) from the participants table and the long-format
segment-wise EEG features table: one row per (subject, segment), indexed by a unique
external participant ID, with one column per (electrode, EEG feature) pair.

Location: src/eeg_data_prep.py (import from the project root as `src.eeg_data_prep`).

Usage — all steps at once (e.g. in the modelling notebook):

    from src.eeg_data_prep import EEGDataPreparer
    prep = EEGDataPreparer(participants_df, eeg_df)
    full_df = prep.run()

Usage — step by step (as in the EDA notebook, section 3):

    prep = EEGDataPreparer(participants_df, eeg_df)
    participants_df_with_ext_id = prep.prepare_participants()   # 3.2
    eeg_df_wide_version = prep.prepare_eeg()                    # 3.3
    prep.validate_wide_columns()                                # 3.4
    full_df = prep.merge()                                      # 3.5
    prep.merge_summary()                                        # 3.5 - check the merge
"""
import numpy as np
import pandas as pd

EXTERNAL_PARTICIPANT_ID_COLUMN_NAME = 'external_participant_id'


# ---------------------------------------------------------------------------
# Helper functions (section 3.1)
# ---------------------------------------------------------------------------
def create_external_participant_id(df: pd.DataFrame, lab_column_name: str, internal_participant_id_column_name: str):
    df[EXTERNAL_PARTICIPANT_ID_COLUMN_NAME] = (
        df[lab_column_name].apply(lambda d: d.replace('ds00', '')) +
        df[internal_participant_id_column_name].apply(lambda d: d.replace('sub-', ''))
    ).astype('int64')
    df_without_previous_fields = df.drop([lab_column_name, internal_participant_id_column_name], axis=1)

    return df_without_previous_fields


def split_numerical_by_category(
    df: pd.DataFrame,
    categorical_field: str,
    numerical_fields: list[str],
    group_by_fields: list[str],
    fill_value=np.nan,
    aggfunc: str = "sum",
) -> pd.DataFrame:
    """
    Replaces `categorical_field` and `numerical_fields` with '<category>_<numerical_field>'
    columns and aggregates the result per `group_by_fields`.
    """
    cat_series = df[categorical_field]
    categories = cat_series.dropna().unique()

    masks = {
        cat_val: (cat_series == cat_val).fillna(False).to_numpy(dtype=bool)
        for cat_val in categories
    }

    # Keep non-matching rows as NaN before aggregation so aggfunc (sum, mean, min, etc.) is accurate
    new_cols = {}
    for num_col in numerical_fields:
        num_values = df[num_col].to_numpy()
        for cat_val, mask in masks.items():
            new_cols[f"{cat_val}_{num_col}"] = np.where(mask, num_values, np.nan)

    new_cols_df = pd.DataFrame(new_cols, index=df.index)
    new_col_names = list(new_cols.keys())

    if group_by_fields:
        result = pd.concat([df[group_by_fields], new_cols_df], axis=1)
        grouped = result.groupby(
            group_by_fields,
            dropna=False,
            sort=False,
            as_index=False,
            observed=True,
        )
        if aggfunc == "sum":
            result = grouped.sum(min_count=1)
        else:
            result = grouped.agg(aggfunc)
    else:
        if aggfunc == "sum":
            result = new_cols_df.sum(min_count=1).to_frame().T
        else:
            result = new_cols_df.agg(aggfunc).to_frame().T

    if pd.notna(fill_value) and new_col_names:
        result[new_col_names] = result[new_col_names].fillna(fill_value)

    return result


# ---------------------------------------------------------------------------
# The data-preparation pipeline (sections 3.2 - 3.5)
# ---------------------------------------------------------------------------
class EEGDataPreparer:
    """
    Turns the raw participants table and the raw long-format EEG table into `full_df`.

    The raw DataFrames are copied, so the originals passed in are never modified.

    Attributes filled in by the steps:
        participants_df_with_ext_id   participants table indexed by the external ID       (prepare_participants)
        eeg_recording_duration        the (single) duration of every EEG segment, seconds (prepare_eeg)
        eeg_df_with_ext_id            long EEG table with the external ID                (prepare_eeg)
        eeg_df_wide_version           wide EEG table: one row per (subject, onset)       (prepare_eeg)
        eeg_features_types            the EEG feature types (without electrode prefix)   (validate_wide_columns)
        full_df                       the merged, unified dataset                        (merge)

    merge_summary() prints / returns the size of full_df and the subjects lost in the merge.
    """

    def __init__(self, participants_df: pd.DataFrame, eeg_df: pd.DataFrame):
        self.participants_df = participants_df.copy()
        self.eeg_df = eeg_df.copy()

        self.participants_df_with_ext_id = None
        self.eeg_recording_duration = None
        self.eeg_df_with_ext_id = None
        self.eeg_df_wide_version = None
        self.eeg_features_types = None
        self.full_df = None

    # 3.2 -------------------------------------------------------------------
    def prepare_participants(self) -> pd.DataFrame:
        """Replace dataset + participant_id with the external ID and drop source_subject_id."""
        # Parse participants dataset
        self.participants_df_with_ext_id = create_external_participant_id(
            self.participants_df, lab_column_name='dataset', internal_participant_id_column_name='participant_id'
        ).drop(['source_subject_id'], axis=1).set_index(EXTERNAL_PARTICIPANT_ID_COLUMN_NAME)

        return self.participants_df_with_ext_id

    # 3.3 -------------------------------------------------------------------
    def prepare_eeg(self) -> pd.DataFrame:
        """Check the segment duration is uniform, then pivot the EEG table from long to wide."""
        eeg_df = self.eeg_df

        # Make sure that all EEG rows have the same recording duration
        if len(eeg_df['duration_s'].unique()) != 1:
            raise NotImplementedError(f'The duration must be uniform. Durations list: {eeg_df["duration_s"].unique().tolist()}')
        # Set the only single duration as constant, present it later
        self.eeg_recording_duration = eeg_df['duration_s'].unique()[0]

        # Parse EEG dataset. Create separated feature for each electrode and EEG numerical feature
        self.eeg_df_with_ext_id = create_external_participant_id(
            eeg_df, lab_column_name='dataset', internal_participant_id_column_name='subject'
        ).drop(['group', 'duration_s', 'segment_index'], axis=1)
        self.eeg_df_wide_version = split_numerical_by_category(
            df=self.eeg_df_with_ext_id, categorical_field='electrode',
            numerical_fields=self.eeg_df_with_ext_id.columns.drop([EXTERNAL_PARTICIPANT_ID_COLUMN_NAME, 'onset_s', 'electrode']),
            group_by_fields=[EXTERNAL_PARTICIPANT_ID_COLUMN_NAME, 'onset_s']
        ).set_index(EXTERNAL_PARTICIPANT_ID_COLUMN_NAME)

        return self.eeg_df_wide_version

    # 3.4 -------------------------------------------------------------------
    def validate_wide_columns(self) -> None:
        """Sanity check: number of wide EEG columns == max electrodes per subject x number of EEG feature types."""
        self._require('eeg_df_wide_version', 'prepare_eeg')

        # Sanity check - number of features should be equal to: max{electrodes_num_per_participant} * EEG_features_num
        max_electrodes_num_per_participant = self.eeg_df_with_ext_id.groupby(EXTERNAL_PARTICIPANT_ID_COLUMN_NAME)['electrode'].nunique().max()
        self.eeg_features_types = self.eeg_df_with_ext_id.columns.drop([EXTERNAL_PARTICIPANT_ID_COLUMN_NAME, 'onset_s', 'electrode'])
        eeg_features_full_list = self.eeg_df_wide_version.columns.drop('onset_s')

        if len(eeg_features_full_list) < max_electrodes_num_per_participant * len(self.eeg_features_types):
            raise NotImplementedError('There are some missing features')
        elif len(eeg_features_full_list) > max_electrodes_num_per_participant * len(self.eeg_features_types):
            raise NotImplementedError('There are some extra features')

    # 3.5 -------------------------------------------------------------------
    def merge(self) -> pd.DataFrame:
        """Inner join of the participants table and the wide EEG table on the external ID."""
        self._require('participants_df_with_ext_id', 'prepare_participants')
        self._require('eeg_df_wide_version', 'prepare_eeg')

        # Create the full dataset
        self.full_df = self.participants_df_with_ext_id.merge(
            self.eeg_df_wide_version, left_index=True, right_index=True, how='inner')

        return self.full_df

    def merge_summary(self, verbose: bool = True) -> dict:
        """Check the result of the merge: size of full_df and subjects lost in the inner join.

        Returns a dict with the numbers (and the IDs of the lost subjects); prints them when verbose=True.
        """
        self._require('full_df', 'merge')
        full_df = self.full_df

        not_in_eeg = self.participants_df_with_ext_id.index.difference(full_df.index)
        not_in_participants = self.eeg_df_wide_version.index.unique().difference(full_df.index)

        summary = {
            'segment_duration_s': self.eeg_recording_duration,
            'n_rows': full_df.shape[0],
            'n_columns': full_df.shape[1],
            'n_subjects': full_df.index.nunique(),
            'subjects_without_eeg': not_in_eeg.tolist(),
            'subjects_without_clinical_data': not_in_participants.tolist(),
        }
        if verbose:
            print(f'Segment duration: {self.eeg_recording_duration} s')
            print(f'full_df: {full_df.shape[0]:,} rows (subject x segment) x {full_df.shape[1]} columns')
            print(f'Subjects in full_df: {full_df.index.nunique()}')
            print(f'Subjects lost in the inner join — no EEG: {len(not_in_eeg)}, no clinical data: {len(not_in_participants)}')
        return summary

    # All steps -------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Run sections 3.2 - 3.5 in order and return full_df."""
        self.prepare_participants()
        self.prepare_eeg()
        self.validate_wide_columns()
        return self.merge()

    def _require(self, attribute: str, step: str) -> None:
        if getattr(self, attribute) is None:
            raise RuntimeError(f'Run {step}() first.')
