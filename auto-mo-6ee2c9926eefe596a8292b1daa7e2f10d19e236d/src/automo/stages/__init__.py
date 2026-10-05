"""Pipeline stages.

Import stage classes directly from their modules (e.g.
``from automo.stages.train import TrainingStage``) so that importing the
package does not eagerly pull in the training engine's heavy dependencies.
"""
