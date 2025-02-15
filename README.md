# Meta-Tool

Large language models (LLMs) have showcased remarkable capabilities as autonomous agents when augmented with external tools. Existing benchmarks primarily assess the proficiency of LLMs in utilizing provided tools within specific scenarios. Equipped with fixed tool sets, LLMs struggle with addressing diverse user inquiries in open-world tasks. In open-world function calling, LLMs need to retrieve suitable tools and use them to resolve user's problem. To address the challenging task, we introduce Meta-Tool, a versatile and plug-and-play tool retrieval system designed to unleash the full intelligence of LLMs in function calling. Drawing inspiration from the myriad of enhanced approaches associated with Retrieval-Augmented Generation (RAG), Meta-Tool employs a hypothesize-retrieve-invoke framework. We further propose Meta-Bench, a comprehensive benchmark for evaluating LLMs in open-world function calling and associated tasks. Meta-Bench encompasses $2,800$ dialogues and $7,361$ tools, spanning ten distinct scenarios to provide a robust and diverse testing categories. In conjunction, we present MT-LLaMA, a finetuned version of LLaMA-3.1, which exhibits remarkable performance improvements. Our empirical experiments reveal that Meta-Tool significantly enhances the ability of advanced LLMs to identify and leverage the most pertinent tools, thereby improving response accuracy to user queries. Moreover, fine-tuning enables even smaller-scale LLMs to achieve notable gains in open-world function calling.
