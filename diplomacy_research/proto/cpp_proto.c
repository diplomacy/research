// Source: http://yz.mit.edu/wp/fast-native-c-protocol-buffers-from-python/
#include <Python.h>

static PyMethodDef CppProtoMethods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef cppprotomodule = {
    PyModuleDef_HEAD_INIT,
    "cpp_proto",  /* name of module */
    "",           /* module documentation, may be NULL */
    -1,           /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    CppProtoMethods
};

PyMODINIT_FUNC
PyInit_cpp_proto(void)
{
    return PyModule_Create(&cppprotomodule);
}
