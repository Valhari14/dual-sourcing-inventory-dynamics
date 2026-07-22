Deployment
==========

Docker
------

This project can be deployed using :code:`docker compose`. Please run the following commands in the project root directory for:

- Tests

.. code-block:: console

    docker compose run tests

- Jupyter lab on the `localhost:8888`

.. code-block:: console

    docker compose up juplab

- Streamlit application, with interactive training session and results dashboard:

.. code-block:: console

    docker compose up app

In case changes are introduced in the code or the build pipeline, you can run the above commands by adding the :code:`--build` option, e.g. :code:`docker run tests --build`.

Web Application
---------------

The current web application allows the user to fit models based on uniform or custom demand.
To do so, the user needs to define a demand distribution or upload relevant demand data, then choose the dual-sourcing model parameters and finally setup the neural network architecture and training parameters.
Once the user completes these steps, the resulting plots are presented in the `Results` tab. This process is visualized in the following video:

.. video:: ../_static/app_vid.mp4
    :width: 100%
