# ai_engine/operator/__init__.py

def get_operator(config: dict, dry_run: bool = False):
    from ai_engine.operator.operator import VeloxOperator
    return VeloxOperator(config, dry_run=dry_run)


__all__ = ["get_operator"]