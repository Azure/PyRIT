# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from pathlib import Path
from typing import Literal, Optional

# import pycountry
from iso3166 import countries

from pyrit.datasets.dataset_helper import FILE_TYPE_HANDLERS, fetch_examples
from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_seclists_bias_testing_dataset(
    source: str = (
        "https://raw.githubusercontent.com/danielmiessler/SecLists/4e747a4/Ai/LLM_Testing/Bias_Testing/"
        "nationality_geographic_bias.txt"
    ),
    source_type: Literal["public_url"] = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
    random_seed: Optional[int] = None,
    country: Optional[str] = None,
    region: Optional[str] = None,
    nationality: Optional[str] = None,
    gender: Optional[str] = None,
    skin_color: Optional[str] = None,
) -> SeedPromptDataset:
    """
    Fetch SecLists AI LLM Bias Testing examples from a specified source and create a SeedPromptDataset.

    Args:
        source (str): The source from which to fetch examples. Defaults to the SecLists repository Bias_Testing.
        source_type (Literal["public_url"]): The type of source ('public_url').
        cache (bool): Whether to cache the fetched examples. Defaults to True.
        data_home (Optional[Path]): Directory to store cached data. Defaults to None.
        random_seed (Optional[int]): Seed for random number generation for reproducibility. Defaults to None.
        country (Optional[str]): Specific country to use for the placeholder. Defaults to None.
        region (Optional[str]): Specific region to use for the placeholder. Defaults to None.
        nationality (Optional[str]): Specific nationality to use for the placeholder. Defaults to None.
        gender (Optional[str]): Specific gender to use for the placeholder. Defaults to None.
        skin_color (Optional[str]): Specific skin color to use for the placeholder. Defaults to None.

    Returns:
        SeedPromptDataset: A SeedPromptDataset containing the examples with placeholders replaced.
    """

    if random_seed is not None:
        random.seed(random_seed)

    # Determine the file type from the source URL
    file_type = source.split(".")[-1]
    if file_type not in FILE_TYPE_HANDLERS:
        valid_types = ", ".join(FILE_TYPE_HANDLERS.keys())
        raise ValueError(f"Invalid file_type. Expected one of: {valid_types}.")

    # Fetch the examples using the provided `fetch_examples` function
    examples = fetch_examples(source, source_type, cache, data_home)

    # NEW:
    # Build country names list from iso3166
    country_names = [c.name for c in countries]

    # Build subdivisions list from all countries that have subdivisions

    subdivision_names = [
        "California",
        "Texas",
        "New York",
        "Florida",
        "Illinois",
        "Pennsylvania",
        "Ohio",
        "Georgia",
        "North Carolina",
        "Michigan",
        "New Jersey",
        "Virginia",
        "Washington",
        "Arizona",
        "Massachusetts",
        "Tennessee",
        "Indiana",
        "Missouri",
        "Maryland",
        "Wisconsin",
        "Colorado",
        "Minnesota",
        "South Carolina",
        "Alabama",
        "Louisiana",
        "Kentucky",
        "Oregon",
        "Oklahoma",
        "Connecticut",
        "Utah",
        "Iowa",
        "Nevada",
        "Arkansas",
        "Mississippi",
        "Kansas",
        "New Mexico",
        "Nebraska",
        "West Virginia",
        "Idaho",
        "Hawaii",
        "New Hampshire",
        "Maine",
        "Montana",
        "Rhode Island",
        "Delaware",
        "South Dakota",
        "North Dakota",
        "Alaska",
        "Vermont",
        "Wyoming",
        "Ontario",
        "Quebec",
        "British Columbia",
        "Alberta",
        "Manitoba",
        "Saskatchewan",
        "Nova Scotia",
        "New Brunswick",
        "Newfoundland and Labrador",
        "Prince Edward Island",
        "Northwest Territories",
        "Yukon",
        "Nunavut",
        "England",
        "Scotland",
        "Wales",
        "Northern Ireland",
        "Greater London",
        "Bavaria",
        "Baden-Wurttemberg",
        "Berlin",
        "Brandenburg",
        "Bremen",
        "Hamburg",
        "Hesse",
        "Lower Saxony",
        "North Rhine-Westphalia",
        "Rhineland-Palatinate",
        "Saarland",
        "Saxony",
        "Saxony-Anhalt",
        "Schleswig-Holstein",
        "Thuringia",
        "Ile-de-France",
        "Provence-Alpes-Cote d'Azur",
        "Auvergne-Rhone-Alpes",
        "Nouvelle-Aquitaine",
        "Occitanie",
        "Grand Est",
        "Hauts-de-France",
        "Normandie",
        "Bretagne",
        "Centre-Val de Loire",
        "Bourgogne-Franche-Comte",
        "Pays de la Loire",
        "Corse",
        "New South Wales",
        "Victoria",
        "Queensland",
        "Western Australia",
        "South Australia",
        "Tasmania",
        "Australian Capital Territory",
        "Northern Territory",
        "Tokyo",
        "Osaka",
        "Kanagawa",
        "Aichi",
        "Saitama",
        "Chiba",
        "Hyogo",
        "Hokkaido",
        "Fukuoka",
        "Kyoto",
        "Shizuoka",
        "Hiroshima",
        "Ibaraki",
        "Sao Paulo",
        "Rio de Janeiro",
        "Minas Gerais",
        "Bahia",
        "Rio Grande do Sul",
        "Parana",
        "Pernambuco",
        "Ceara",
        "Para",
        "Santa Catarina",
        "Goias",
        "Maranhao",
        "Paraiba",
        "Espirito Santo",
        "Mato Grosso",
        "Amazonas",
        "Maharashtra",
        "Uttar Pradesh",
        "Tamil Nadu",
        "West Bengal",
        "Karnataka",
        "Gujarat",
        "Rajasthan",
        "Andhra Pradesh",
        "Madhya Pradesh",
        "Kerala",
        "Delhi",
        "Bihar",
        "Telangana",
        "Odisha",
        "Assam",
        "Punjab",
        "Haryana",
        "Beijing",
        "Shanghai",
        "Guangdong",
        "Jiangsu",
        "Zhejiang",
        "Shandong",
        "Henan",
        "Sichuan",
        "Hubei",
        "Hunan",
        "Hebei",
        "Fujian",
        "Anhui",
        "Liaoning",
        "Shaanxi",
        "Jiangxi",
        "Heilongjiang",
        "Guangxi",
        "Yunnan",
        "Moscow",
        "Saint Petersburg",
        "Sverdlovsk Oblast",
        "Novosibirsk Oblast",
        "Republic of Tatarstan",
        "Nizhny Novgorod Oblast",
        "Chelyabinsk Oblast",
        "Samara Oblast",
        "Rostov Oblast",
        "Republic of Bashkortostan",
        "Krasnoyarsk Krai",
        "Lombardy",
        "Lazio",
        "Campania",
        "Sicily",
        "Veneto",
        "Emilia-Romagna",
        "Piedmont",
        "Apulia",
        "Tuscany",
        "Calabria",
        "Sardinia",
        "Liguria",
        "Madrid",
        "Catalonia",
        "Andalusia",
        "Valencia",
        "Galicia",
        "Castile and Leon",
        "Basque Country",
        "Castilla-La Mancha",
        "Canary Islands",
        "Murcia",
    ]

    # Helper classes to maintain compatibility with the existing lambda functions
    class CountryCompat:
        def __init__(self, name):
            self.name = name

    class SubdivisionCompat:
        def __init__(self, name):
            self.name = name

    # Create compatibility objects
    countries_compat = [CountryCompat(name) for name in country_names]
    subdivisions_compat = [SubdivisionCompat(name) for name in subdivision_names]

    placeholder_values = {
        "Country": lambda: country if country else random.choice(countries_compat).name,
        "Region": lambda: region if region else random.choice(subdivisions_compat).name,
        "Nationality": lambda: (
            nationality if nationality else random.choice(countries_compat).name
        ),  # Same as Country, the target LLM should be able to understand the context
        "GENDER": lambda: gender if gender else random.choice(["male", "female", "non-binary"]),
        "SKIN_COLOR": lambda: skin_color if skin_color else random.choice(["white", "black", "brown", "yellow"]),
    }

    # Replace placeholders in the examples with generated values
    filled_examples = []
    for example in examples:
        prompt = example["prompt"]
        for placeholder, generator in placeholder_values.items():
            values_used = set()
            while f"[{placeholder}]" in prompt:
                value = generator()
                # Ensure the new value is not the same as the previous one
                while value in values_used:
                    value = generator()
                values_used.add(value)
                prompt = prompt.replace(f"[{placeholder}]", value, 1)

        filled_examples.append(prompt)

    # Create SeedPrompt instances from each example in 'filled_examples'
    seed_prompts = [
        SeedPrompt(
            value=example,
            data_type="text",
            name="SecLists Bias Testing Examples",
            dataset_name="SecLists Bias Testing Examples",
            harm_categories=["bias_testing"],
            description="A dataset of SecLists AI LLM Bias Testing examples with placeholders replaced.",
        )
        for example in filled_examples
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)

    return seed_prompt_dataset
