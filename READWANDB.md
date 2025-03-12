# W&B Quickstart
[docs.wandb.ai](https://docs.wandb.ai/quickstart/)

> W&B Quickstart

W&B Quickstart

  3 minute read  

Install W&B and start tracking your machine learning experiments in minutes.

Sign up and create an API key[](#sign-up-and-create-an-api-key)
---------------------------------------------------------------

An API key authenticates your machine to W&B. You can generate an API key from your user profile.

For a more streamlined approach, you can generate an API key by going directly to [https://wandb.ai/authorize](https://wandb.ai/authorize). Copy the displayed API key and save it in a secure location such as a password manager.

1.  Click your user profile icon in the upper right corner.
2.  Select **User Settings**, then scroll to the **API Keys** section.
3.  Click **Reveal**. Copy the displayed API key. To hide the API key, reload the page.

Install the `wandb` library and log in[](#install-the-wandb-library-and-log-in)
-------------------------------------------------------------------------------

To install the `wandb` library locally and log in:

1.  Set the `WANDB_API_KEY` [environment variable](https://docs.wandb.ai/guides/track/environment-variables/) to your API key.
    
2.  Install the `wandb` library and log in.
    

Start a run and track hyperparameters[](#start-a-run-and-track-hyperparameters)
-------------------------------------------------------------------------------

Initialize a W&B Run object in your Python script or notebook with [`wandb.init()`](https://docs.wandb.ai/ref/python/run/) and pass a dictionary to the `config` parameter with key-value pairs of hyperparameter names and values:

A [run](https://docs.wandb.ai/guides/runs/) is the basic building block of W&B. You will use them often to [track metrics](https://docs.wandb.ai/guides/track/), [create logs](https://docs.wandb.ai/guides/artifacts/), and more.

Put it all together[](#put-it-all-together)
-------------------------------------------

Putting it all together, your training script might look similar to the following code example.

That’s it. Navigate to the W&B App at [https://wandb.ai/home](https://wandb.ai/home) to view how the metrics we logged with W&B (accuracy and loss) improved during each training step.

![](https://docs.wandb.ai/images/quickstart/quickstart_image.png)

The image above (click to expand) shows the loss and accuracy that was tracked from each time we ran the script above. Each run object that was created is show within the **Runs** column. Each run name is randomly generated.

What’s next?[](#whats-next)
---------------------------

Explore the rest of the W&B ecosystem.

1.  Check out [W&B Integrations](https://docs.wandb.ai/guides/integrations/) to learn how to integrate W&B with your ML framework such as PyTorch, ML library such as Hugging Face, or ML service such as SageMaker.
2.  Organize runs, embed and automate visualizations, describe your findings, and share updates with collaborators with [W&B Reports](https://docs.wandb.ai/guides/reports/).
3.  Create [W&B Artifacts](https://docs.wandb.ai/guides/artifacts/) to track datasets, models, dependencies, and results through each step of your machine learning pipeline.
4.  Automate hyperparameter search and explore the space of possible models with [W&B Sweeps](https://docs.wandb.ai/guides/sweeps/).
5.  Understand your datasets, visualize model predictions, and share insights in a [central dashboard](https://docs.wandb.ai/guides/models/tables/).
6.  Navigate to W&B AI Academy and learn about LLMs, MLOps and W&B Models from hands-on [courses](https://wandb.me/courses).


Last modified March 7, 2025: [dcacf6f](https://github.com/wandb/docs/commit/dcacf6f907eb1d79b60d9f73df126a0ee37ae94f)