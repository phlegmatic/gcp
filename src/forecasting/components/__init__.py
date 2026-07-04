"""KFP component definitions.

Every reusable pipeline step lives here as a `@component`-decorated function.
Rules for this package (enforced by AI_AGENT_GUIDELINES.md):
  * Each component pins the SMALLEST possible `packages_to_install` list.
  * Never import heavy libs at module top-level -- import inside the function so
    the KFP compiler does not require them in the authoring environment.
  * Components exchange data ONLY via KFP Artifacts / parameters, never globals.
"""
