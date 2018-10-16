
// IMAGE_process_part1Dlg.h: 헤더 파일
//

#pragma once


// CIMAGEprocesspart1Dlg 대화 상자
class CIMAGEprocesspart1Dlg : public CDialogEx
{
// 생성입니다.
public:
	CIMAGEprocesspart1Dlg(CWnd* pParent = nullptr);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_IMAGE_PROCESS_PART1_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	void Basic_image(unsigned char *src_image, int width, int height, boolean save = false);
	CImage x_image;
	CImage y_image;
	CImage x_hist_image;
	CImage y_hist_image;
	double x_HIST[256];
	CString result;
	unsigned char clamping(int pixel);
	int ArithmeticPixel(unsigned char *src_image, int width, int height,
		double meticPixel, CString mode);
	int LogicPixel(unsigned char *src_image, int width, int height,
		CString mode);
	int GammaCorrection(unsigned char *src_image, int width, int height,
		double gamma_val);
	unsigned char* _toPixel(unsigned char* src_image, int image_size);
	int Save_image(CString name);
	int Create_image(CImage& image, unsigned char* src_image, int width, int height, CString name, boolean save = false);
	//int Create_y_image(unsigned char* src_image, int width, int height, CString name, boolean save = false);
	int Negertive_invert(unsigned char* src_image, int width, int height);
	int Posterizing(unsigned char* src_image, int width, int height,
		int post_value);
	int Binarization(unsigned char* src_image, int width, int height, int binary_value);
	int Highright_range(unsigned char* src_image, int widht, int height, int range_min, int range_max);
	int Sampling(unsigned char* src_image, CString mode, int width, int height, int sampling_val);
	int Histogram_graph(CImage& image, unsigned char* src_image, int m_width, int m_height, boolean print = false);
	int Histogram_process(unsigned char* src_image, int width, int height,
		CString mode, int low, int high);
};
