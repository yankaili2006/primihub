#include "gtest/gtest.h"
#include "src/primihub/primitive/oprf/oprf.h"
#include <cstring>

using namespace primihub::oprf;

TEST(oprf_test, sender_key_generation) {
  OprfSender sender;
  EXPECT_NE(sender.key().sk, nullptr);
  EXPECT_NE(sender.key().pk, nullptr);
  EXPECT_NE(sender.key().group, nullptr);
}

TEST(oprf_test, receiver_blind) {
  OprfReceiver receiver;
  std::vector<uint8_t> input = {1, 2, 3, 4, 5};
  auto blinded = receiver.Blind(input);
  EXPECT_FALSE(blinded.empty());
  EXPECT_NE(blinded.size(), 0);
}

TEST(oprf_test, full_oprf_protocol) {
  OprfSender sender;
  OprfReceiver receiver;

  std::vector<uint8_t> input = {0x48, 0x65, 0x6c, 0x6c, 0x6f};  // "Hello"

  // Step 1: Receiver blinds the input
  auto blinded = receiver.Blind(input);
  EXPECT_FALSE(blinded.empty());

  // Step 2: Sender evaluates the blinded input
  auto evaluated = sender.Evaluate(blinded);
  EXPECT_FALSE(evaluated.empty());

  // Step 3: Receiver finalizes (unblinds) to get PRF output
  auto output = receiver.Finalize(input, evaluated, receiver.r());
  EXPECT_EQ(output.size(), 32);
}

TEST(oprf_test, consistency) {
  // Same input + same key = same output (deterministic)
  OprfSender sender;

  std::vector<uint8_t> input_a = {0x48, 0x65, 0x6c, 0x6c, 0x6f};  // "Hello"
  std::vector<uint8_t> input_b = {0x57, 0x6f, 0x72, 0x6c, 0x64};  // "World"

  // Evaluate both inputs directly (without blinding)
  auto output_a1 = sender.BlindEvaluate(input_a);
  auto output_a2 = sender.BlindEvaluate(input_a);
  auto output_b = sender.BlindEvaluate(input_b);

  // Same input → same output
  EXPECT_EQ(output_a1, output_a2);

  // Different input → different output
  EXPECT_NE(output_a1, output_b);
}

TEST(oprf_test, empty_input) {
  OprfSender sender;
  auto result = sender.Evaluate({});
  EXPECT_TRUE(result.empty());
}
