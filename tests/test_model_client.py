from app.services.model_client import ModelClient


class FakeModelClient(ModelClient):
    def generate(
        self,
        model_name: str,
        inputs: list[str],
    ) -> list[str]:
        return [
            f"generated: {item}"
            for item in inputs
        ]


def test_model_client_generates_outputs() -> None:
    client = FakeModelClient()

    outputs = client.generate(
        model_name="test-model",
        inputs=[
            "Question 1",
            "Question 2",
        ],
    )

    assert outputs == [
        "generated: Question 1",
        "generated: Question 2",
    ]