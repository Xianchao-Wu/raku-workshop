#!/usr/bin/env python
# coding: utf-8

# # 🎨 Data Designer Tutorial: The Basics
# 
# #### 📚 What you'll learn
# 
# This notebook demonstrates the basics of Data Designer by generating a simple product review dataset.
# 

# ### 📦 Import Data Designer
# 
# - `data_designer.config` provides access to the configuration API.
# 
# - `DataDesigner` is the main interface for data generation.
# 

# In[1]:


import data_designer.config as dd
from data_designer.interface import DataDesigner


# In[1]:


import data_designer.config as dd
from data_designer.interface import DataDesigner


# ### ⚙️ Initialize the Data Designer interface
# 
# - `DataDesigner` is the main object responsible for managing the data generation process.
# 
# - When initialized without arguments, the [default model providers](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/models/default-model-settings/) are used.
# 

# In[2]:


data_designer = DataDesigner()


# ### 🎛️ Define model configurations
# 
# - Each `ModelConfig` defines a model that can be used during the generation process.
# 
# - The "model alias" is used to reference the model in the Data Designer config (as we will see below).
# 
# - The "model provider" is the external service that hosts the model (see the [model config](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/models/default-model-settings/) docs for more details).
# 
# - By default, we use [build.nvidia.com](https://build.nvidia.com/models) as the model provider.
# 

# In[3]:


# This name is set in the model provider configuration.
MODEL_PROVIDER = "nvidia"

# The model ID is from build.nvidia.com.
MODEL_ID = "nvidia/nemotron-3-nano-30b-a3b"

# We choose this alias to be descriptive for our use case.
MODEL_ALIAS = "nemotron-nano-v3"

model_configs = [
    dd.ModelConfig(
        alias=MODEL_ALIAS,
        model=MODEL_ID,
        provider=MODEL_PROVIDER,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    )
]


# ### 🏗️ Initialize the Data Designer Config Builder
# 
# - The Data Designer config defines the dataset schema and generation process.
# 
# - The config builder provides an intuitive interface for building this configuration.
# 
# - The list of model configs is provided to the builder at initialization.
# 

# In[4]:


config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)


# ## 🎲 Getting started with sampler columns
# 
# - Sampler columns offer non-LLM based generation of synthetic data.
# 
# - They are particularly useful for **steering the diversity** of the generated data, as we demonstrate below.
# 
# <br>
# 
# You can view available samplers using the config builder's `info` property:
# 

# In[5]:


config_builder.info.display("samplers")


# Let's start designing our product review dataset by adding product category and subcategory columns.
# 

# In[6]:


config_builder.add_column(
    dd.SamplerColumnConfig(
        name="product_category",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=[
                "Electronics",
                "Clothing",
                "Home & Kitchen",
                "Books",
                "Home Office",
            ],
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="product_subcategory",
        sampler_type=dd.SamplerType.SUBCATEGORY,
        params=dd.SubcategorySamplerParams(
            category="product_category",
            values={
                "Electronics": [
                    "Smartphones",
                    "Laptops",
                    "Headphones",
                    "Cameras",
                    "Accessories",
                ],
                "Clothing": [
                    "Men's Clothing",
                    "Women's Clothing",
                    "Winter Coats",
                    "Activewear",
                    "Accessories",
                ],
                "Home & Kitchen": [
                    "Appliances",
                    "Cookware",
                    "Furniture",
                    "Decor",
                    "Organization",
                ],
                "Books": [
                    "Fiction",
                    "Non-Fiction",
                    "Self-Help",
                    "Textbooks",
                    "Classics",
                ],
                "Home Office": [
                    "Desks",
                    "Chairs",
                    "Storage",
                    "Office Supplies",
                    "Lighting",
                ],
            },
        ),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="target_age_range",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=["18-25", "25-35", "35-50", "50-65", "65+"]),
    )
)

# Optionally validate that the columns are configured correctly.
data_designer.validate(config_builder)


# Next, let's add samplers to generate data related to the customer and their review.
# 

# In[7]:


config_builder.add_column(
    dd.SamplerColumnConfig(
        name="customer",
        sampler_type=dd.SamplerType.PERSON_FROM_FAKER,
        params=dd.PersonFromFakerSamplerParams(age_range=[18, 70], locale="en_US"),
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="number_of_stars",
        sampler_type=dd.SamplerType.UNIFORM,
        params=dd.UniformSamplerParams(low=1, high=5),
        convert_to="int",  # Convert the sampled float to an integer.
    )
)

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="review_style",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["rambling", "brief", "detailed", "structured with bullet points"],
            weights=[1, 2, 2, 1],
        ),
    )
)

data_designer.validate(config_builder)


# ## 🦜 LLM-generated columns
# 
# - The real power of Data Designer comes from leveraging LLMs to generate text, code, and structured data.
# 
# - When prompting the LLM, we can use Jinja templating to reference other columns in the dataset.
# 
# - As we see below, nested json fields can be accessed using dot notation.
# 

# In[8]:


config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="product_name",
        prompt=(
            "You are a helpful assistant that generates product names. DO NOT add quotes around the product name.\n\n"
            "Come up with a creative product name for a product in the '{{ product_category }}' category, focusing "
            "on products related to '{{ product_subcategory }}'. The target age range of the ideal customer is "
            "{{ target_age_range }} years old. Respond with only the product name, no other text."
        ),
        model_alias=MODEL_ALIAS,
    )
)

config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="customer_review",
        prompt=(
            "You are a customer named {{ customer.first_name }} from {{ customer.city }}, {{ customer.state }}. "
            "You are {{ customer.age }} years old and recently purchased a product called {{ product_name }}. "
            "Write a review of this product, which you gave a rating of {{ number_of_stars }} stars. "
            "The style of the review should be '{{ review_style }}'. "
            "Respond with only the review, no other text."
        ),
        model_alias=MODEL_ALIAS,
    )
)

data_designer.validate(config_builder)


# ### 🔁 Iteration is key – preview the dataset!
# 
# 1. Use the `preview` method to generate a sample of records quickly.
# 
# 2. Inspect the results for quality and format issues.
# 
# 3. Adjust column configurations, prompts, or parameters as needed.
# 
# 4. Re-run the preview until satisfied.
# 

# In[9]:


preview = data_designer.preview(config_builder, num_records=2)


# In[10]:


# Run this cell multiple times to cycle through the 2 preview records.
preview.display_sample_record()


# In[11]:


# The preview dataset is available as a pandas DataFrame.
print(preview.dataset)


# ### 📊 Analyze the generated data
# 
# - Data Designer automatically generates a basic statistical analysis of the generated data.
# 
# - This analysis is available via the `analysis` property of generation result objects.
# 

# In[12]:


# Print the analysis as a table.
preview.analysis.to_report()


# ### 🆙 Scale up!
# 
# - Happy with your preview data?
# 
# - Use the `create` method to submit larger Data Designer generation jobs.
# 

# In[13]:


results = data_designer.create(config_builder, num_records=10, dataset_name="tutorial-1")


# In[14]:


# Load the generated dataset as a pandas DataFrame.
dataset = results.load_dataset()

dataset.head()


# In[15]:


# Load the analysis results into memory.
analysis = results.load_analysis()

analysis.to_report()


# ## ⏭️ Next Steps
# 
# Now that you've seen the basics of Data Designer, check out the following notebooks to learn more about:
# 
# - [Structured outputs and jinja expressions](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/2-structured-outputs-and-jinja-expressions/)
# 
# - [Seeding synthetic data generation with an external dataset](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/3-seeding-with-a-dataset/)
# 
# - [Providing images as context](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/4-providing-images-as-context/)
# 
# - [Generating images](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/5-generating-images/)
# 
