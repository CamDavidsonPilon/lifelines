## Contributing to lifelines


### Questions about survival analysis?
If you are using lifelines for survival analysis and have a question about "how do I do X?" or "what does Y do?", the best place to ask that is either in our [discussion channel](https://github.com/camdavidsonpilon/lifelines/discussions) or at [stats.stackexchange.com](https://stats.stackexchange.com/).


### Submitting bugs or other errors observed

We appreciate all bug reports submitted, as this will help the entire community get a better product. Please open up an issue in the Github Repository. If possible, please provide a code snippet, and what version of lifelines you are using.


### Submitting new feature requests

Please open up an issue in the Github Repository with as much context as possible about the feature you would like to see. Also useful is to link to other libraries/software that have that feature.


### Submitting code, or other changes

If you are interested in contributing to lifelines (and we thank you for the interest!), we recommend first opening up an issue in the GitHub repository to discuss the changes. From there, we can together plan how to execute the changes. See the Development section below for how to setup a local environment.

## Development

### Setting up a lifelines development environment

1. From the root directory of `lifelines` activate your [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) (if you plan to use one).
2. Install the development requirements and [`pre-commit`](https://pre-commit.com) hooks. If you are on Mac, Linux, or [Windows `WSL`](https://docs.microsoft.com/en-us/windows/wsl/faq) you can use the provided [`Makefile`](https://github.com/CamDavidsonPilon/lifelines/blob/master/Makefile). Just type `make` into the console and you're ready to start developing. This will also install the dev-requirements.

### Formatting

`lifelines` uses the [`black`](https://github.com/ambv/black) python formatter.
There are 3 different ways to format your code.
1. Use the [`Makefile`](https://github.com/CamDavidsonPilon/lifelines/blob/master/Makefile).
   * `make lint`
2. Call `black` directly and pass the correct line length.
   * `black . -l 120`
3. Have your code formatted automatically during commit with the `pre-commit` hook.
   * Stage and commit your unformatted changes: `git commit -m "your_commit_message"`
   * Code that needs to be formatted will "fail" the commit hooks and be formatted for you.
   * Stage the newly formatted python code: `git add *.py`
   * Recall your original commit command and commit again: `git commit -m "your_commit_message"`

### Running the tests

You can optionally run the test suite after install with

    py.test
