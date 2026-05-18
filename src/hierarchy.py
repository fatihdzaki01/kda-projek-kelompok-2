"""
Generalization Hierarchy for QI Attributes.

Contains multi-level generalization functions and the GeneralizationHierarchy class.
"""


def generalize_age(value, level):
    """
    Age Multi-Level Hierarchy

    Original values: 1-13 (age groups)
    Level 0: Original (1, 2, 3, ..., 13)
    Level 1: Young (1-3), Adult (4-7), MiddleAge (8-10), Senior (11-13)
    Level 2: Young-Adult (1-7), Senior (8-13)
    Level 3: Any
    """
    if level == 0:
        return value
    elif level == 1:
        if value <= 3:
            return 'Young'
        elif value <= 7:
            return 'Adult'
        elif value <= 10:
            return 'MiddleAge'
        else:
            return 'Senior'
    elif level == 2:
        if value <= 7:
            return 'Young-Adult'
        else:
            return 'Senior'
    elif level == 3:
        return 'Any'
    else:
        raise ValueError(f"Invalid level {level}. Must be 0-3.")


def generalize_bmi(value, level):
    """
    BMI Multi-Level Hierarchy

    Original values: 12-98 (continuous BMI)
    Level 0: Original value
    Level 1: Underweight (<18.5), Normal (18.5-25), Overweight (25-30), Obese (30+)
    Level 2: Normal-Range (<25), Overweight-Range (25+)
    Level 3: Any
    """
    if level == 0:
        return value
    elif level == 1:
        if value < 18.5:
            return 'Underweight'
        elif value < 25:
            return 'Normal'
        elif value < 30:
            return 'Overweight'
        else:
            return 'Obese'
    elif level == 2:
        if value < 25:
            return 'Normal-Range'
        else:
            return 'Overweight-Range'
    elif level == 3:
        return 'Any'
    else:
        raise ValueError(f"Invalid level {level}. Must be 0-3.")


def generalize_income(value, level):
    """
    Income Multi-Level Hierarchy

    Original values: 1-8 (income brackets)
    Level 0: Original (1-8)
    Level 1: Low (1-3), Middle (4-6), High (7-8)
    Level 2: Low-Middle (1-6), High (7-8)
    Level 3: Any
    """
    if level == 0:
        return value
    elif level == 1:
        if value <= 3:
            return 'Low'
        elif value <= 6:
            return 'Middle'
        else:
            return 'High'
    elif level == 2:
        if value <= 6:
            return 'Low-Middle'
        else:
            return 'High'
    elif level == 3:
        return 'Any'
    else:
        raise ValueError(f"Invalid level {level}. Must be 0-3.")


def generalize_education(value, level):
    """
    Education Multi-Level Hierarchy

    Original values: 1-6 (education levels)
    Level 0: Original (1-6)
    Level 1: LowEdu (1-2), MidEdu (3-4), HighEdu (5-6)
    Level 2: LowEdu (1-4), HighEdu (5-6)
    Level 3: Any
    """
    if level == 0:
        return value
    elif level == 1:
        if value <= 2:
            return 'LowEdu'
        elif value <= 4:
            return 'MidEdu'
        else:
            return 'HighEdu'
    elif level == 2:
        if value <= 4:
            return 'LowEdu'
        else:
            return 'HighEdu'
    elif level == 3:
        return 'Any'
    else:
        raise ValueError(f"Invalid level {level}. Must be 0-3.")


def generalize_genhlth(value, level):
    """
    General Health Multi-Level Hierarchy

    Original values: 1-5 (health rating)
    Level 0: Original (1-5)
    Level 1: Good (1-2), Fair (3), Poor (4-5)
    Level 2: Good (1-3), Poor (4-5)
    Level 3: Any
    """
    if level == 0:
        return value
    elif level == 1:
        if value <= 2:
            return 'Good'
        elif value == 3:
            return 'Fair'
        else:
            return 'Poor'
    elif level == 2:
        if value <= 3:
            return 'Good'
        else:
            return 'Poor'
    elif level == 3:
        return 'Any'
    else:
        raise ValueError(f"Invalid level {level}. Must be 0-3.")


def generalize_sex(value, level):
    """
    Sex Multi-Level Hierarchy

    Original values: 0 (Female), 1 (Male)
    Level 0: Original (0, 1)
    Level 1: Female/Male (labeled)
    Level 2: Any

    Note: Sex only has 3 levels (binary attribute)
    """
    if level == 0:
        return value
    elif level == 1:
        return 'Female' if value == 0 else 'Male'
    elif level >= 2:
        return 'Any'
    else:
        raise ValueError(f"Invalid level {level}. Must be 0-2.")


class GeneralizationHierarchy:
    """
    Multi-level generalization hierarchy for all QI attributes.

    Provides unified interface: generalize(attribute, value, level)
    """

    def __init__(self):
        self.max_levels = {
            'Age': 3,
            'BMI': 3,
            'Income': 3,
            'Education': 3,
            'GenHlth': 3,
            'Sex': 2,
        }

    def generalize(self, attribute, value, level):
        """
        Generalize a value to specified level.
        """
        if attribute not in self.max_levels:
            raise ValueError(f"Unknown attribute: {attribute}")

        if level < 0 or level > self.max_levels[attribute]:
            raise ValueError(
                f"Invalid level {level} for {attribute}. "
                f"Must be 0-{self.max_levels[attribute]}."
            )

        if attribute == 'Age':
            return generalize_age(value, level)
        elif attribute == 'BMI':
            return generalize_bmi(value, level)
        elif attribute == 'Income':
            return generalize_income(value, level)
        elif attribute == 'Education':
            return generalize_education(value, level)
        elif attribute == 'GenHlth':
            return generalize_genhlth(value, level)
        elif attribute == 'Sex':
            return generalize_sex(value, level)
        else:
            raise ValueError(f"Unknown attribute: {attribute}")

    def get_max_level(self, attribute):
        """Get maximum generalization level for attribute."""
        return self.max_levels.get(attribute, 3)

    def apply_to_dataframe(self, df, levels_dict):
        """
        Apply generalization to entire dataframe.

        Args:
            df (pd.DataFrame): Input dataframe
            levels_dict (dict): {attribute: level} mapping

        Returns:
            pd.DataFrame: Generalized dataframe
        """
        df_gen = df.copy()

        for attr, level in levels_dict.items():
            if attr in df_gen.columns:
                df_gen[attr] = df_gen[attr].apply(
                    lambda x: self.generalize(attr, x, level)
                )

        return df_gen


# Instantiate global hierarchy
HIERARCHY = GeneralizationHierarchy()
