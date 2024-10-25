import pika
import json

class RabbitMQClient:
    def __init__(self, host='localhost', port=5672, username='guest', password='guest'):
        self.connection_params = pika.ConnectionParameters(
            host=host,
            port=port,
            credentials=pika.PlainCredentials(username, password)
        )
        self.connection = None
        self.channel = None

    def connect(self):
        """Establishes a connection to RabbitMQ and opens a channel."""
        if not self.connection or self.connection.is_closed:
            self.connection = pika.BlockingConnection(self.connection_params)
            self.channel = self.connection.channel()

    def declare_exchange(self, exchange_name, exchange_type='topic'):
        """Declares an exchange with the specified name and type."""
        if not self.channel:
            self.connect()
        self.channel.exchange_declare(exchange=exchange_name, exchange_type=exchange_type)

    def declare_queue(self, queue_name):
        """Declares a queue with the specified name."""
        if not self.channel:
            self.connect()
        self.channel.queue_declare(queue=queue_name)

    def bind_queue(self, queue_name, exchange_name, routing_key):
        """Binds a queue to an exchange with a specific routing key."""
        if not self.channel:
            self.connect()
        self.channel.queue_bind(queue=queue_name, exchange=exchange_name, routing_key=routing_key)

    def send_message(self, exchange_name, routing_key, message):
        """Sends a JSON message to a specified exchange and routing key."""
        if not self.channel:
            self.connect()
        if not isinstance(message, dict):
            raise ValueError("Message must be a dictionary.")
        self.channel.basic_publish(
            exchange=exchange_name,
            routing_key=routing_key,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                content_type='application/json'
            )
        )
        print(f"Sent message to {exchange_name} with routing key {routing_key}: {message}")

    def receive_messages(self, queue_name, callback):
        """Receives messages from a specified queue and processes them using the callback."""
        if not self.channel:
            self.connect()
        def on_message(ch, method, properties, body):
            message = json.loads(body)
            callback(message)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        self.channel.basic_consume(queue=queue_name, on_message_callback=on_message)
        print(f"Waiting for messages in queue {queue_name}. To exit press CTRL+C")
        self.channel.start_consuming()

    def close(self):
        """Closes the connection to RabbitMQ."""
        if self.connection and not self.connection.is_closed:
            self
