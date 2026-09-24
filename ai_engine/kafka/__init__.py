# ai_engine/kafka/__init__.py
# Lazy imports only — prevents import errors if confluent-kafka not installed

def get_producer(config):
    from ai_engine.kafka.producer import VeloxProducer
    return VeloxProducer(config)


def get_consumer(config, group_id, topics):
    from ai_engine.kafka.consumer import VeloxConsumer
    return VeloxConsumer(config, group_id, topics)


def get_bridge(config=None):
    from ai_engine.kafka.bridge_config import BridgeConfig
    from ai_engine.kafka.kafka_prometheus_bridge import KafkaPrometheusBridge
    if config is None:
        config = BridgeConfig.from_yaml()
    return KafkaPrometheusBridge(config)


__all__ = ["get_producer", "get_consumer", "get_bridge"]