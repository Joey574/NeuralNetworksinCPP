
//
//fmat ReLU(fmat ATotal) {
//
//	return ATotal.for_each([](fmat::elem_type& val) {val = val < 0.0f ? 0.0f : val; });
//}
//
//float ReLUDerivative(float A) {
//	return A > 0.0f ? 1.0f : 0.0f;
//}
//
//fmat Softmax(fmat ATotal) {
//	return 0;
//}