### Getting Started:

1. First install [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) for convenience.
2. Activate the base environment and create a new environment:
    ```conda create --name mmp python=3.10```
3. Activate env: ```conda activate mmp```
4. Cd into package and install the package: ```cd PATH_TO_PACKAGE/manipulator_motion_planning && pip install -e .```
5. Verify your installation: ```python -c "import manipulator_motion_planning; print(manipulator_motion_planning.__file__)"```
6. Run the client: ```python3 ./scripts/run_simulation_client.py```
7. Run pick and place: ```python3 ./scripts/run_pick_and_place.py```

You should see the following simulation:
![demo](output.gif)