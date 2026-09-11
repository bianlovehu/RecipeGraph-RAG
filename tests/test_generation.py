import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document

from rag_modules.generation_integration import GenerationIntegrationModule


def completion(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
    )


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.module = GenerationIntegrationModule.__new__(GenerationIntegrationModule)
        self.module.model_name = "deepseek-v4-flash"
        self.module.temperature = 0.1
        self.module.max_tokens = 2048
        self.module.client = Mock()

    def test_answer_generation_disables_thinking(self):
        self.module.client.chat.completions.create.return_value = completion("番茄和鸡蛋。")

        answer = self.module.generate_adaptive_answer(
            "番茄炒蛋需要什么？",
            [Document(page_content="番茄炒蛋需要番茄和鸡蛋。")],
        )

        self.assertEqual(answer, "番茄和鸡蛋。")
        self.assertEqual(
            self.module.client.chat.completions.create.call_args.kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_empty_answer_is_retried(self):
        self.module.client.chat.completions.create.side_effect = [
            completion(""),
            completion("第二次生成成功。"),
        ]

        answer = self.module.generate_adaptive_answer(
            "推荐一道菜",
            [Document(page_content="凉拌鸡丝是一道低卡菜。")],
        )

        self.assertEqual(answer, "第二次生成成功。")
        self.assertEqual(self.module.client.chat.completions.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
